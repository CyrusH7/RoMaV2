import json
from itertools import combinations

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
    doc["images"]["a.png"]["camera_to_world"][0][0] = -1
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="Invalid rigid"):
        load_priors(path, ["a.png"])


def test_full_prior_pipeline(tmp_path):
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
        pose[:3, :3] = Rotation.from_rotvec(rng.normal(0, 0.01, 3)).as_matrix()
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
        num_threads=1,
    )
    assert max(m["registered_images"] for m in result["models"]) == 5
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
    assert (out / "sparse/0/prior_refinement.json").exists()
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

    def partial_matcher(a, b):
        if missing in (a.name, b.name):
            return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
        xy_a, xy_b, scores = matcher(a, b)
        if (a.name, b.name) == (names[0], names[1]):
            return (
                np.vstack((xy_a, [40, 40])),
                np.vstack((xy_b, [40, 435])),
                np.append(scores, 1),
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
        (bad_idx,) = np.flatnonzero(np.all(kp == [40, 40], axis=1))
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
