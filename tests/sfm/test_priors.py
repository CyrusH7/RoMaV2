import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pycolmap
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from romav2.sfm.ablation import trajectory_metrics
from romav2.sfm.colmap import reconstruct
from romav2.sfm.priors import Priors, load_priors


def test_prior_validation(tmp_path):
    doc = {
        "pose_convention": "camera_to_world_opencv",
        "units": "meters",
        "rig_calibration": {"depth_from_rgb": np.eye(4).tolist()},
        "images": {
            "a.png": {
                "intrinsics": [500, 500, 320, 240],
                "camera_to_world": np.eye(4).tolist(),
            }
        },
    }
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(doc))
    loaded = load_priors(path, ["a.png"])
    assert np.allclose(loaded.poses["a.png"], np.eye(4))
    assert loaded.rig_calibration == doc["rig_calibration"]
    doc["images"]["a.png"]["camera_to_world"][0][0] = -1
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="Invalid rigid"):
        load_priors(path, ["a.png"])


@pytest.mark.parametrize(
    ("graph", "rotation_noise", "expected_status"),
    [(False, 0.01, None), (True, 0.01, "kept_input_poses"), (True, 0.03, "refined")],
)
def test_full_prior_pipeline(tmp_path, graph, rotation_noise, expected_status):
    root = tmp_path / "images"
    root.mkdir()
    names = [f"{i}.png" for i in range(5)]
    rng = np.random.default_rng(7)
    xyz = np.column_stack(
        (rng.uniform(-1.2, 1.2, 300), rng.uniform(-1, 1, 300), rng.uniform(4, 7, 300))
    )
    coords, poses, intrinsics = {}, {}, {}
    for i, name in enumerate(names):
        Image.new("RGB", (640, 480)).save(root / name)
        center = np.array([0.6 * (i - 2), 0.05 * (i % 2), 0])
        pose = np.eye(4)
        pose[:3, 3] = center
        poses[name] = pose
        cam = xyz - center
        focal = 500.0 + 10.0 * (i % 2)
        intrinsics[name] = np.array([focal, focal, 320, 240])
        coords[name] = cam[:, :2] / cam[:, 2:] * focal + [320, 240]

    def matcher(a, b):
        return coords[a.name], coords[b.name], np.ones(len(xyz))

    noisy = {n: t.copy() for n, t in poses.items()}
    for pose in noisy.values():
        pose[:3, 3] += rng.normal(0, 0.01, 3)
        pose[:3, :3] = Rotation.from_rotvec(
            rng.normal(0, rotation_noise, 3)
        ).as_matrix()
    priors = Priors(intrinsics, noisy, 0.05, 3.0)
    out = tmp_path / "both"
    result = reconstruct(
        root,
        out,
        names,
        list(combinations(names, 2)),
        matcher,
        priors=priors,
        use_intrinsics_prior=True,
        use_pose_prior=True,
        fix_intrinsics=True,
        prior_iterations=2,
        init_from_poses=graph,
        refine_initial_pose="graph" if graph else False,
        joint_prior_ba=True,
        num_threads=1,
    )
    assert max(m["registered_images"] for m in result["models"]) == 5
    if graph:
        seed = json.loads((out / "initial_pose_refinement.json").read_text())
        assert seed["status"] == expected_status
        if expected_status == "refined":
            assert seed["held_out_score_after_px"] < seed["held_out_score_before_px"]
        else:
            assert seed["reason"] == "raw_matches_support_input_poses"
    m = pycolmap.Reconstruction(str(out / "sparse/0"))
    assert len(m.cameras) == 2
    for image in m.images.values():
        assert np.allclose(m.cameras[image.camera_id].params, intrinsics[image.name])
    estimated = {}
    for im in m.images.values():
        pose = np.eye(4)
        pose[:3, :3] = im.cam_from_world().rotation.matrix().T
        pose[:3, 3] = im.projection_center()
        estimated[im.name] = pose
    metrics = trajectory_metrics(estimated, poses)
    assert metrics["raw_ate_rmse_m"] < 0.08
    assert abs(metrics["sim3_scale_correction"] - 1) < 0.05
    assert (
        json.loads((out / "sparse/0/prior_refinement.json").read_text())["solver"]
        == "joint_metric"
    )
    image = m.find_image_with_name(names[0])
    depth_errors = []
    for point in image.points2D:
        if point.has_point3D():
            index = np.argmin(np.linalg.norm(coords[image.name] - point.xy, axis=1))
            depth_errors.append(m.points3D[point.point3D_id].xyz[2] - xyz[index, 2])
    assert len(depth_errors) > 200
    assert np.sqrt(np.mean(np.square(depth_errors))) < 0.1
    from types import SimpleNamespace

    from romav2.sfm.ablation import evaluate

    document = {
        "images": {n: {"camera_to_world": t.tolist()} for n, t in poses.items()}
    }
    (tmp_path / "reference.json").write_text(json.dumps(document))
    (tmp_path / "priors.json").write_text(json.dumps(document))
    assert not evaluate(SimpleNamespace(output=tmp_path))
    report = json.loads((tmp_path / "metrics.json").read_text())
    assert report["both"]["status"] == "ok"
    assert report["both"]["registered_images"] == 5
    assert report["baseline"]["status"] == "failed_or_incomplete"


def test_evaluation_scale_is_not_hidden():
    reference, doubled = {}, {}
    for i, c in enumerate([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]]):
        pose = np.eye(4)
        pose[:3, 3] = c
        reference[str(i)] = pose
        scaled = pose.copy()
        scaled[:3, 3] *= 2
        doubled[str(i)] = scaled
    metrics = trajectory_metrics(doubled, reference)
    assert metrics["sim3_ate_rmse_m"] < 1e-10
    assert metrics["se3_ate_rmse_m"] > 0.1
    assert metrics["sim3_scale_correction"] == pytest.approx(0.5)


def test_match_cache_reuses_arrays_without_model(tmp_path, monkeypatch):
    import torch

    import romav2
    from romav2.sfm.matching import RomaMatcher

    calls = []

    class FakeModel:
        def __init__(self):
            calls.append("load")

        def apply_setting(self, setting):
            pass

        def match(self, a, b):
            return {}

        def sample(self, pred, count):
            return torch.zeros(1, 4), torch.ones(1), None, None

        def to_pixel_coordinates(self, matches, ha, wa, hb, wb):
            return torch.tensor([[wa / 2, ha / 2]]), torch.tensor([[wb / 2, hb / 2]])

    monkeypatch.setitem(romav2.__dict__, "RoMaV2", FakeModel)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    Image.new("RGB", (20, 20), (20, 0, 0)).save(a)
    Image.new("RGB", (20, 20), (40, 0, 0)).save(b)
    cache = tmp_path / "cache"
    first = RomaMatcher("fast", 100, 0, cache)(a, b)
    second = RomaMatcher("fast", 100, 0, cache)(a, b)
    assert calls == ["load"]
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    RomaMatcher("fast", 50, 0, cache)(a, b)
    assert calls == ["load", "load"]


def test_different_intrinsics_are_not_silently_shared(tmp_path):
    names = ["a.png", "b.png", "c.png"]
    root = tmp_path / "images"
    root.mkdir()
    for name in names:
        Image.new("RGB", (640, 480)).save(root / name)
    ks = {
        names[0]: np.array([500, 500, 320, 240]),
        names[1]: np.array([510, 510, 320, 240]),
        names[2]: np.array([500, 500, 320, 240]),
    }
    priors = Priors(ks, {})

    def empty(a, b):
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)

    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="did not reconstruct"):
        reconstruct(
            root,
            out,
            names,
            list(combinations(names, 2)),
            empty,
            priors=priors,
            use_intrinsics_prior=True,
            fix_intrinsics=True,
            num_threads=1,
        )
    db = pycolmap.Database.open(str(out / "database.db"))
    try:
        camera_ids = {n: db.read_image_with_name(n).camera_id for n in names}
        assert camera_ids["a.png"] == camera_ids["c.png"]
        assert camera_ids["a.png"] != camera_ids["b.png"]
        for name in names:
            assert np.allclose(db.read_camera(camera_ids[name]).params, ks[name])
    finally:
        db.close()


def test_fixed_pose_triangulation_uses_input_world_coordinates(tmp_path):
    from romav2.sfm.priors import _prepare_prior_frame

    images = tmp_path / "images"
    images.mkdir()
    names = [f"view{i}.png" for i in range(5)]
    k = np.array([500.0, 505.0, 320.0, 240.0])
    xyz = np.random.default_rng(123).uniform([-1, -1, 5], [1, 1, 8], (250, 3))
    poses, keypoints = {}, {}
    for i, name in enumerate(names):
        Image.new("RGB", (640, 480)).save(images / name)
        pose = np.eye(4)
        pose[:3, 3] = [(i - 2) * 0.45, 0.15 * (i % 2), 0.1 * (i % 3)]
        pose[:3, :3] = Rotation.from_euler("y", (2 - i) * 3, degrees=True).as_matrix()
        cam_xyz = (xyz - pose[:3, 3]) @ pose[:3, :3]
        keypoints[name] = cam_xyz[:, :2] / cam_xyz[:, 2:] * k[:2] + k[2:]
        poses[name] = pose

    def matcher(a, b):
        return keypoints[a.name], keypoints[b.name], np.ones(len(xyz))

    priors = Priors({name: k for name in names}, poses)
    out = tmp_path / "fixed"
    reconstruct(
        images,
        out,
        names,
        list(combinations(names, 2)),
        matcher,
        priors=priors,
        use_intrinsics_prior=True,
        fix_intrinsics=True,
        fix_poses=True,
        num_threads=1,
    )
    m = pycolmap.Reconstruction(str(out / "sparse/0"))
    assert m.num_points3D() > 200 and m.num_reg_images() == len(names)
    assert 0 <= m.compute_mean_reprojection_error() < 0.01
    recovered = np.stack([point.xyz for point in m.points3D.values()])
    nearest = np.linalg.norm(recovered[:, None] - xyz, axis=2).min(axis=1)
    assert np.percentile(nearest, 95) < 1e-4
    for im in m.images.values():
        np.testing.assert_allclose(
            im.cam_from_world().inverse().matrix(), poses[im.name][:3], atol=1e-10
        )

    # The initialization experiment must not count a supplied pose without
    # visual evidence, and the other cameras must pass through soft refinement.
    missing = "unobserved.png"
    Image.new("RGB", (640, 480)).save(images / missing)
    all_names = names + [missing]
    all_priors = Priors({n: k for n in all_names}, {**poses, missing: np.eye(4)})
    behind = {}
    for name in names[:2]:
        pose = poses[name]
        camera_xyz = (np.array([0.1, 1.6, -6.0]) - pose[:3, 3]) @ pose[:3, :3]
        behind[name] = camera_xyz[:2] / camera_xyz[2] * k[:2] + k[2:]

    def partial_matcher(a, b):
        if missing in (a.name, b.name):
            return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
        xy_a, xy_b, scores = matcher(a, b)
        if (a.name, b.name) == (names[0], names[1]):
            return (
                np.vstack((xy_a, [40, 40], behind[a.name])),
                np.vstack((xy_b, [40, 435], behind[b.name])),
                np.append(scores, [1, 1]),
            )
        return xy_a, xy_b, scores

    seeded = tmp_path / "seeded"
    seeded_result = reconstruct(
        images,
        seeded,
        all_names,
        list(combinations(all_names, 2)),
        partial_matcher,
        priors=all_priors,
        use_intrinsics_prior=True,
        fix_intrinsics=True,
        init_from_poses=True,
        num_threads=1,
        prior_iterations=1,
    )
    with pycolmap.Database.open(str(seeded / "database.db")) as db:
        a, b = [db.read_image_with_name(n).image_id for n in names[:2]]
        kp = db.read_keypoints(a)[:, :2]
        for bad_xy in ([40, 40], behind[names[0]]):
            (bad_idx,) = np.flatnonzero(np.all(np.isclose(kp, bad_xy), axis=1))
            assert bad_idx in db.read_matches(a, b)[:, 0]
            assert bad_idx not in db.read_two_view_geometry(a, b).inlier_matches[:, 0]
    assert seeded_result["models"][0]["registered_images"] == len(names)
    info = json.loads((seeded / "sparse/0/prior_refinement.json").read_text())
    assert info["initialization"] == "input_poses"
    assert info["status"] == "refined_in_prior_frame"
    recovered_model = pycolmap.Reconstruction(str(seeded / "sparse/0"))
    recovered = np.stack([pt.xyz for pt in recovered_model.points3D.values()])
    assert (
        np.percentile(np.linalg.norm(recovered[:, None] - xyz, axis=2).min(axis=1), 95)
        < 1e-4
    )

    # Biased orientation measurements must not rotate centers that already agree.
    biased = {n: p.copy() for n, p in poses.items()}
    for p in biased.values():
        p[:3, :3] = Rotation.from_euler("z", 5, degrees=True).as_matrix() @ p[:3, :3]
    assert _prepare_prior_frame(m, Priors({}, biased)) == ("preserved", 1.0)
    # Millimeter-confidence inputs can have centimeter-scale *initial* visual errors.
    precise = {n: p.copy() for n, p in poses.items()}
    for i, p in enumerate(precise.values()):
        p[2, 3] += (-1) ** i * 0.008
    initial = pycolmap.Reconstruction(str(out / "sparse/0"))
    _prepare_prior_frame(initial, Priors({}, precise, position_sigma=0.001))
    residuals = [
        np.linalg.norm(im.projection_center() - precise[im.name][:3, 3])
        for im in initial.images.values()
    ]
    assert np.median(residuals) < 0.01

    # The wider search radius must still recover a coherent offset of similar size.
    initial = pycolmap.Reconstruction(str(out / "sparse/0"))
    initial.transform(pycolmap.Sim3d(1.0, pycolmap.Rotation3d(), [0.008, 0, 0]))
    action, _ = _prepare_prior_frame(initial, Priors({}, poses, position_sigma=0.001))
    assert action == "recovered_from_centers"
    for im in initial.images.values():
        np.testing.assert_allclose(
            im.projection_center(), poses[im.name][:3, 3], atol=1e-8
        )
    incompatible = {n: pose.copy() for n, pose in poses.items()}
    for i, pose in enumerate(incompatible.values()):
        pose[:3, 3] = [i**2, (-1) ** i, i**3]
    with pytest.raises(ValueError, match="Cannot establish a metric frame"):
        _prepare_prior_frame(m, Priors({}, incompatible, position_sigma=0.001))

    m.transform(pycolmap.Sim3d(2.0, pycolmap.Rotation3d(), [5.0, 3.0, 2.0]))
    action, scale = _prepare_prior_frame(m, priors)
    assert action == "recovered_from_centers" and scale == pytest.approx(0.5)
    for im in m.images.values():
        np.testing.assert_allclose(
            im.projection_center(), poses[im.name][:3, 3], atol=1e-8
        )

    two_views = tmp_path / "fixed_two"
    reconstruct(
        images,
        two_views,
        names[:2],
        [(names[0], names[1])],
        matcher,
        priors=priors,
        use_intrinsics_prior=True,
        fix_intrinsics=True,
        fix_poses=True,
        num_threads=1,
    )
    assert pycolmap.Reconstruction(str(two_views / "sparse/0")).num_points3D() > 200

    # Each disjoint feature group appears in exactly two cameras of a larger scene.
    edges = list(zip(names, names[1:]))

    def pair_only(a, b):
        edge = edges.index((a.name, b.name))
        indices = slice(50 * edge, 50 * (edge + 1))
        return keypoints[a.name][indices], keypoints[b.name][indices], np.ones(50)

    chain = tmp_path / "known_baselines"
    reconstruct(
        images,
        chain,
        names,
        edges,
        pair_only,
        priors=priors,
        cell_size=0.01,
        use_intrinsics_prior=True,
        fix_intrinsics=True,
        init_from_poses=True,
        prior_iterations=1,
        num_threads=1,
    )
    chain_model = pycolmap.Reconstruction(str(chain / "sparse/0"))
    assert chain_model.num_reg_images() == len(names)
    assert all(im.num_points3D >= 6 for im in chain_model.images.values())
    assert all(p.track.length() == 2 for p in chain_model.points3D.values())


def test_adjacent_metrics_ignore_gaps():
    reference = {}
    for i in range(5):
        pose = np.eye(4)
        pose[0, 3] = i
        reference[str(i)] = pose
    poses = {n: reference[n] for n in ("0", "2", "4")}
    metrics = trajectory_metrics(poses, reference)
    assert metrics["adjacent_rpe_pairs"] == 0
    assert metrics["adjacent_rpe_rotation_mean_deg"] is None
    poses["1"] = reference["1"]
    assert trajectory_metrics(poses, reference)["adjacent_rpe_pairs"] == 2


def test_prior_registration_retries_against_map_and_rejects_unsupported_poses(tmp_path):
    names = [f"v{i}.png" for i in range(7)]
    images = tmp_path / "images"
    images.mkdir()
    k = np.array([500.0, 500.0, 320.0, 240.0])
    xyz = np.random.default_rng(729).uniform([-1, -1, 4], [1, 1, 6], (120, 3))
    poses, xy = {}, {}
    for i, name in enumerate(names):
        Image.new("RGB", (640, 480)).save(images / name)
        center = np.array([(i - 2) * 0.24, 0.0, 0.0])
        local = xyz - center
        xy[name] = k[:2] * local[:, :2] / local[:, 2:] + k[2:]
        pose = np.eye(4)
        pose[:3, 3] = center
        if i >= 4:
            pose[:3, :3] = Rotation.from_euler(
                "x", 30 if i == 5 else 6, degrees=True
            ).as_matrix()
        poses[name] = pose

    def matcher(a, b):
        return xy[a.name], xy[b.name], np.ones(len(xyz))

    pairs = (
        list(combinations(names[:4], 2))
        + [(a, b) for a in names[:4] for b in names[4:6]]
        + [(names[0], names[6])]
    )
    out = tmp_path / "model"
    reconstruct(
        images,
        out,
        names,
        pairs,
        matcher,
        priors=Priors(dict.fromkeys(names, k), poses),
        use_intrinsics_prior=True,
        fix_intrinsics=True,
        init_from_poses=True,
        num_threads=1,
        prior_iterations=1,
        cell_size=0.01,
    )
    diagnostic = json.loads((out / "sparse/0/prior_registration.json").read_text())
    assert diagnostic[names[4]]["status"] == "registered"
    assert diagnostic[names[5]]["status"] == "outside_prior_uncertainty"
    assert diagnostic[names[6]]["status"] == "insufficient_reference_geometry"
    model = pycolmap.Reconstruction(str(out / "sparse/0"))
    assert model.num_reg_images() == 5
    view = model.find_image_with_name(names[4])
    assert view.num_points3D == diagnostic[names[4]]["inliers"]
    np.testing.assert_allclose(
        view.projection_center(), poses[names[4]][:3, 3], atol=1e-3
    )
    np.testing.assert_allclose(
        view.cam_from_world().rotation.matrix(), np.eye(3), atol=1e-4
    )


def test_initial_sampson_refines_orientation_without_moving_centers(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    names = ["a.png", "b.png", "c.png", "d.png"]
    rng = np.random.default_rng(7)
    xyz = rng.uniform([-0.8, -0.6, 3.5], [0.8, 0.65, 5.5], (160, 3))
    intrinsics = {}
    input_poses = {}
    pixels = {}
    for i, name in enumerate(names):
        Image.new("RGB", (640, 480)).save(images / name)
        k = np.array([530.0 + 10 * i, 460.0, 320.0, 240.0])
        intrinsics[name] = k
        pose = np.eye(4)
        pose[:3, 3] = [i * 0.24, 0.08 * (i % 2), 0.04 * i]
        local = xyz - pose[:3, 3]
        pixels[name] = local[:, :2] / local[:, 2:] * k[:2] + k[2:]
        if i == 1:
            pose[:3, :3] = Rotation.from_euler("x", 1.4, degrees=True).as_matrix()
        input_poses[name] = pose
    priors = Priors(intrinsics, input_poses)
    pairs = list(combinations(names, 2))

    def matcher(a, b):
        return pixels[a.name], pixels[b.name], np.ones(len(xyz))

    def run(path, priors, mode=True):
        return reconstruct(
            images,
            path,
            names,
            pairs,
            matcher,
            priors=priors,
            use_intrinsics_prior=True,
            fix_intrinsics=True,
            init_from_poses=True,
            refine_initial_pose=mode,
            prior_iterations=1,
            num_threads=1,
            cell_size=0.01,
        )

    out = tmp_path / "refined"
    original = {n: pose.copy() for n, pose in input_poses.items()}
    result = run(out, priors)
    report = json.loads((out / "initial_pose_refinement.json").read_text())
    selected = report["selected"]
    assert report["status"] == "refined"
    assert report["translation_locked"] is True
    assert selected["refined_image"] == "b.png"
    updated = np.asarray(report["camera_to_world"])
    np.testing.assert_allclose(updated[:3, 3], original["b.png"][:3, 3], atol=1e-12)
    assert selected["center_shift_m"] == pytest.approx(0.0, abs=1e-12)
    assert np.degrees(Rotation.from_matrix(updated[:3, :3]).magnitude()) < 0.35
    assert (
        selected["held_out_median_px_after"] < selected["held_out_median_px_before"] / 4
    )
    assert selected["third_view_tracks"] >= 20
    assert (
        selected["third_view_clipped_px_after"]
        < selected["third_view_clipped_px_before"]
    )
    assert selected["refined_baseline_m"] == pytest.approx(
        selected["baseline_m"], abs=1e-12
    )
    assert result["models"][0]["registered_images"] == len(names)
    snapshot = json.loads((out / "priors_used.json").read_text())
    for n in names:
        np.testing.assert_array_equal(priors.poses[n], original[n])
        np.testing.assert_array_equal(
            snapshot["images"][n]["camera_to_world"], original[n]
        )
    final = json.loads((out / "sparse/0/prior_refinement.json").read_text())
    assert final["initialization"] == "sampson_lm_fixed_centers"

    from dataclasses import replace

    perturbed = {n: pose.copy() for n, pose in original.items()}
    perturbed["b.png"][:3, 3] += [0.0, 0.024, -0.008]
    joint_output = tmp_path / "joint"
    run(joint_output, replace(priors, poses=perturbed), mode="joint")
    report = json.loads((joint_output / "initial_pose_refinement.json").read_text())
    assert report["status"] == "refined"
    assert report["mode"] == "joint"
    update = report["selected"]
    assert update["refined_image"] == "b.png"
    estimate = np.asarray(update["camera_to_world"])
    assert np.degrees(Rotation.from_matrix(estimate[:3, :3]).magnitude()) < 1.4
    assert np.linalg.norm(estimate[:3, 3] - original["b.png"][:3, 3]) < np.linalg.norm(
        perturbed["b.png"][:3, 3] - original["b.png"][:3, 3]
    )
    ba = json.loads((joint_output / "sparse/0/prior_refinement.json").read_text())
    assert ba["initialization"] == "sampson_lm_joint"

    skipped = tmp_path / "skipped"
    run(skipped, replace(priors, position_sigma=1.0))
    report = json.loads((skipped / "initial_pose_refinement.json").read_text())
    assert report["status"] == "kept_input_poses"


def test_refined_map_falls_back_if_visual_connectivity_drops(tmp_path, monkeypatch):
    from romav2.sfm import colmap
    from romav2.sfm.io import write_json
    from romav2.sfm.tracks import track_component_sizes

    names = list(range(4))
    first = [[0, 1], [1, 2], [2, 3]]
    second = [[0, 1], [2, 3]]
    original_priors, refined_priors = object(), object()
    db_path = tmp_path / "database.db"
    sparse = tmp_path / "sparse"
    sparse.mkdir()

    class Model:
        def __init__(self, tracks):
            from types import SimpleNamespace

            self.points3D = {
                i: SimpleNamespace(
                    track=SimpleNamespace(
                        elements=[SimpleNamespace(image_id=n) for n in track]
                    )
                )
                for i, track in enumerate(tracks)
            }

        def reg_image_ids(self):
            return names

        def write(self, path):
            write_json(Path(path) / "model.json", {"tracks": len(self.points3D)})

    def verify(priors, db):
        from sqlite3 import connect

        with connect(db) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS geometry (prior TEXT)")
            conn.execute(
                "INSERT INTO geometry VALUES (?)",
                ("original" if priors is original_priors else "refined",),
            )
        return priors

    def triangulate(db, images, sparse, priors, options):
        tracks = first if priors is original_priors else second
        write_json(sparse.parent / "prior_triangulation.json", {"tracks": len(tracks)})
        return {0: Model(tracks)}

    monkeypatch.setattr(colmap, "_triangulate_known_poses", triangulate)
    report = {}
    models, stats = colmap._validate_refined_map(
        db_path, tmp_path, sparse, original_priors, refined_priors, None, verify, report
    )
    assert stats is original_priors
    assert track_component_sizes(models[0]) == [4]
    assert report["reason"] == "refined_map_lost_connectivity"
    from sqlite3 import connect

    with connect(db_path) as conn:
        assert conn.execute("SELECT prior FROM geometry").fetchall() == [("original",)]


def test_joint_metric_baseline_needs_consistent_precise_reference_views():
    from romav2.sfm.initialization import _metric_baseline

    seed_length, true_length = 0.5, 0.535
    xyz = np.random.default_rng(11).uniform([-0.8, -0.4, 3.0], [0.8, 0.4, 5.0], (64, 3))

    def rays(points):
        return points / points[:, 2:]

    def pose(center):
        return pycolmap.Rigid3d(pycolmap.Rotation3d(), -np.asarray(center))

    obs = (rays(xyz), rays(xyz - [true_length, 0, 0]))
    tracks = [
        (pose([x, 0, 0]), np.arange(len(xyz)), rays(xyz - [x, 0, 0])[:, :2], np.ones(2))
        for x in (-3.0, 3.0, 4.0)
    ]
    args = (pose([0, 0, 0]), pose([seed_length, 0, 0]), obs)
    measured, count = _metric_baseline(*args, tracks, 0.03, 0.0002, 20)
    assert count == 3
    assert seed_length < measured <= true_length + 1e-12
    assert _metric_baseline(*args, tracks[:2], 0.03, 0.0002, 20)[0] == pytest.approx(
        seed_length
    )
    assert _metric_baseline(*args, tracks, 0.03, 0.10, 20)[0] == pytest.approx(
        seed_length
    )


def test_landmark_seed_uses_external_parallax_and_preserves_metric_midpoint():
    from romav2.sfm.seed_landmarks import refine_landmark_pair

    rng = np.random.default_rng(45)
    xyz = rng.uniform([-1, -0.7, 1.5], [1, 0.7, 8], size=(160, 3))
    focal = np.array([510.0, 540.0])

    def camera(x, degrees=0):
        r = Rotation.from_euler("x", degrees, degrees=True).as_matrix()
        return pycolmap.Rigid3d(pycolmap.Rotation3d(r), -r @ np.array([x, 0, 0]))

    def rays(cam):
        points = cam * xyz
        return points / points[:, 2:]

    truth = (camera(-0.25), camera(0.25))
    inputs = (camera(-0.23, 1.15), camera(0.23, -1.15))
    obs = (*map(rays, truth), focal, focal)
    indices = np.arange(len(xyz))

    def tracks(centers):
        return [(cam, indices, rays(cam)[:, :2], focal) for cam in map(camera, centers)]

    def solve(poses, centers):
        return refine_landmark_pair(
            poses, obs, tracks(centers), indices[::4], 0.035, 0.05, 4, 20
        )[0]

    refs = [-2.0, -1.2, -0.6, 0.8, 1.1, 2.0]
    matrices = solve(inputs, refs)
    assert matrices is not None
    a, b = (pycolmap.Rigid3d(m[:3]).inverse() for m in matrices)
    target = truth[1] * truth[0].inverse()
    initial = inputs[1] * inputs[0].inverse()
    updated = b * a.inverse()
    assert (
        updated.rotation * target.rotation.inverse()
    ).angle() < 0.2 * initial.rotation.angle()
    assert np.linalg.norm(
        updated.translation - target.translation
    ) < 0.5 * np.linalg.norm(initial.translation - target.translation)
    np.testing.assert_allclose(matrices[0][:3, 3] + matrices[1][:3, 3], 0, atol=1e-12)
    assert solve(truth, refs) is None
    assert solve(inputs, refs[:2]) is None
    assert solve(inputs, [1.0, 1.0, 1.0]) is None


def test_final_refinement_filters_low_parallax_at_zero_reprojection_error(monkeypatch):
    from romav2.sfm import priors as module

    options = pycolmap.SyntheticDatasetOptions()
    options.num_rigs, options.num_frames_per_rig, options.num_points3D = 1, 3, 2
    model = pycolmap.synthesize_dataset(options)
    near, far = sorted(model.points3D)
    model.points3D[near].xyz = [0, 0, 5]
    model.points3D[far].xyz = [0, 0, 1000]
    poses = {}
    for i, image in enumerate(model.images.values()):
        pose = np.eye(4)
        pose[0, 3] = i
        poses[image.name] = pose
        transform = pycolmap.Rigid3d(pose[:3]).inverse()
        model.frames[image.frame_id].set_cam_from_world(image.camera_id, transform)
        for obs in image.points2D:
            if obs.has_point3D():
                obs.xy = image.camera.img_from_cam(
                    transform * model.points3D[obs.point3D_id].xyz
                )
    monkeypatch.setattr(module, "_alternate_refinement", lambda *args: None)
    report = module.refine_with_pose_priors(model, Priors({}, poses))
    assert list(model.points3D) == [near]
    assert report["points_removed_after_refinement"] == 1
