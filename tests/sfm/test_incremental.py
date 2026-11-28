import json

import numpy as np
import pycolmap
from PIL import Image

from romav2.sfm.colmap import reconstruct
from romav2.sfm.priors import Priors


def test_incremental_registration_uses_only_existing_map_and_keeps_metric_units(
    tmp_path,
):
    images = tmp_path / "images"
    images.mkdir()
    names = [f"{i}.png" for i in range(8)]
    rng = np.random.default_rng(42)
    xyz = rng.uniform([-0.65, -0.75, 4], [0.65, 0.75, 6], (180, 3))
    camera = np.array([510, 500, 320, 240])
    coords, poses = {}, {}
    for i, name in enumerate(names):
        Image.new("RGB", (640, 480)).save(images / name)
        pose = np.eye(4)
        pose[:3, 3] = [0.34 * (i - 3), 0.035 * (i % 2), 0]
        local = xyz - pose[:3, 3]
        coords[name] = local[:, :2] / local[:, 2:] * camera[:2] + camera[2:]
        pose[1, 3] += 0.006 * (-1) ** i
        poses[name] = pose

    def matcher(a, b):
        if b.name == names[-1]:
            return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
        return coords[a.name], coords[b.name], np.ones(len(xyz))

    pairs = [(n, m) for i, n in enumerate(names) for m in names[i + 1 : i + 3]]
    estimates = []
    for scale in (1, 2):
        output = tmp_path / f"out-{scale}"
        scaled = {name: pose.copy() for name, pose in poses.items()}
        for pose in scaled.values():
            pose[:3, 3] *= scale
        reconstruct(
            images,
            output,
            names,
            pairs,
            matcher,
            priors=Priors(dict.fromkeys(names, camera), scaled, 0.02 * scale, 1),
            use_intrinsics_prior=True,
            fix_intrinsics=True,
            incremental_prior=True,
            prior_iterations=1,
            num_threads=1,
            seed=5,
        )
        report = json.loads((output / "incremental_registration.json").read_text())
        order = report["order"]
        model = pycolmap.Reconstruction(str(output / "sparse/0"))
        assert (
            set(order)
            == {im.name for im in model.images.values() if im.has_pose}
            == set(names[:-1])
        )
        assert report["frames"][names[-1]]["status"] != "registered"
        assert set(report["frames"][order[2]]["references"]) <= set(order[:2])
        for j, name in enumerate(order[2:], 2):
            detail = report["frames"][name]
            assert detail["status"] == "registered"
            assert set(detail["references"]) <= set(order[:j])
            assert detail["held_inliers"] >= 3 and detail["evaluations"] >= 1
        assert any(
            set(report["frames"][name]["references"]).isdisjoint(order[:2])
            for name in order[2:]
        )
        assert (
            json.loads((output / "sparse/0/prior_refinement.json").read_text())[
                "alignment_scale"
            ]
            == 1
        )
        estimates.append(
            np.stack(
                [model.find_image_with_name(n).projection_center() for n in names[:-1]]
            )
        )
    np.testing.assert_allclose(estimates[1], 2 * estimates[0], atol=2e-4)


def test_pose_only_ba_corrects_metric_pose_and_rejects_inconsistent_geometry(
    monkeypatch,
):
    from types import SimpleNamespace

    from romav2.sfm.incremental import _localize
    from romav2.sfm.options import PipelineOptions

    camera = pycolmap.Camera(
        model="PINHOLE", width=640, height=480, params=[500, 500, 320, 240]
    )
    rng = np.random.default_rng(3)
    points = rng.uniform([-1.5, -1, 3], [1.5, 1, 6], (200, 3))
    xyz = [SimpleNamespace(xyz=p) for p in points]
    model = SimpleNamespace(point3D=xyz.__getitem__)
    query = SimpleNamespace(camera=camera, image_id=7)
    pixels = camera.img_from_cam(points)
    edges = [(i, i, p) for i, p in enumerate(pixels)]
    start = pycolmap.Rigid3d(
        pycolmap.Rotation3d(np.array([0.008, 0, 0])), np.array([0.02, 0.01, 0.005])
    )
    priors = SimpleNamespace(position_sigma=0.05, rotation_sigma_deg=3)
    options = PipelineOptions()
    result, detail = _localize(model, query, edges, start, priors, options)
    assert detail["status"] == "registered"
    pose, _ = result
    assert pose.rotation.angle() < start.rotation.angle()
    assert np.linalg.norm(pose.inverse().translation) < np.linalg.norm(
        start.inverse().translation
    )

    # The solver must still consider the other start when a PnP fit is rejected.
    import romav2.sfm.incremental as incremental

    least_squares = incremental.least_squares

    def failed_pnp_fit(residual, initial, **kwargs):
        if np.linalg.norm(initial) > 0:
            return SimpleNamespace(x=np.full(6, np.nan), nfev=1)
        return least_squares(residual, initial, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(incremental, "least_squares", failed_pnp_fit)
        result, detail = _localize(model, query, edges, start, priors, options)
    assert result is not None
    assert detail["selected_start"] == "initial_pose"
    assert detail["trials"]["pnp"]["status"] == "nonfinite_pose"
    np.testing.assert_allclose(
        detail["ba_start_camera_to_world"], start.inverse().matrix()
    )

    # Shuffle associations within the held-out pixels, preserving the image split.
    _, groups = np.unique(np.floor(pixels / 48), axis=0, return_inverse=True)
    held_tiles = np.random.default_rng(query.image_id).permutation(groups.max() + 1)[
        ::4
    ]
    held = np.flatnonzero(np.isin(groups, held_tiles))
    for i, j in zip(held, np.roll(held, 1)):
        edges[i] = (i, i, pixels[j])
    rejected, detail = _localize(model, query, edges, start, priors, options)
    assert rejected is None and detail["status"] == "pose_validation_failed"


def test_local_warp_requires_nearby_verified_samples():
    from romav2.sfm.incremental import _transfer_pixels

    samples = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [100, 100]])
    mapped = samples @ np.array([[1, 0.2], [-0.1, 1]]) + [4, 2]
    xy = np.array([[4, 6], [60, 60], [-1, 3]])
    uv, valid = _transfer_pixels(samples, mapped, xy)
    assert valid.tolist() == [True, False, False]
    back, valid = _transfer_pixels(mapped, samples, uv[:1])
    assert valid.all()
    np.testing.assert_allclose(back, xy[:1])
