import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pycolmap
from PIL import Image

path = Path(__file__).resolve().parents[2] / "scripts" / "ablate_chair.py"
spec = importlib.util.spec_from_file_location("ablate_chair", path)
chair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chair)


def test_native_sift_replay_preserves_colmap_feature_indices(tmp_path):
    names = ["a.png", "b.png"]
    for name in names:
        Image.new("RGB", (64, 64)).save(tmp_path / name)
    path = str(tmp_path / "native.db")
    with pycolmap.Database.open(path):
        pass
    pycolmap.import_images(path, str(tmp_path), image_names=names)
    xy = np.array([[10.1, 10.2], [10.2, 10.3], [21, 31]], dtype=np.float32)
    descriptors = np.zeros((3, 128), dtype=np.uint8)
    descriptors[:, 0] = [1, 100, 230]
    raw = np.array([[2, 1], [0, 0]], dtype=np.uint32)
    with pycolmap.Database.open(path) as db:
        ids = [db.read_image_with_name(name).image_id for name in names]
        for i in ids:
            db.write_keypoints(i, xy)
            db.write_descriptors(i, descriptors)
        db.write_matches(*ids, raw)
    pair = tuple(reversed(names))
    store, counts = chair.IndexedSiftMatches(path).load_indexed_matches(
        names, [pair], 0
    )
    assert counts == {"b.png a.png": 2}
    np.testing.assert_array_equal(store.keypoints(names[0]), xy)
    np.testing.assert_array_equal(store.matches[pair], raw[:, ::-1])
    np.testing.assert_allclose(store.scores[pair], [1 / 131, 1])


def test_common_frame_flags_exclude_missing_and_unsupported():
    by = {
        "control_visual": {
            "a": {"observations": 6, "flagged": True},
            "b": {"observations": 9, "flagged": True},
        },
        "pose_visual": {
            "a": {"observations": 7, "flagged": False},
            "b": {"observations": 2, "flagged": False},
        },
    }
    (row,) = chair.comparisons(list(by), by)
    assert (row["common"], row["before_only"], row["corrected"]) == (1, 1, 1)
    assert row["newly_flagged"] == 0


def test_shared_3d_tracks_identify_separate_visual_components():
    tracks = [[1, 2], [2, 3], [4, 5]]
    model = SimpleNamespace(
        reg_image_ids=lambda: range(1, 7),
        points3D={
            i: SimpleNamespace(
                track=SimpleNamespace(
                    elements=[SimpleNamespace(image_id=j) for j in track]
                )
            )
            for i, track in enumerate(tracks)
        },
    )
    assert chair.track_component_sizes(model) == [3, 2, 1]


def test_depth_metric_uses_scaled_camera_z_and_colmap_pixel_origin(tmp_path):
    path = tmp_path / "depth.png"
    Image.fromarray(np.array([[1000, 0], [500, 1500]], dtype=np.uint16)).save(path)
    points = [
        SimpleNamespace(xy=[x, y], point3D_id=i, has_point3D=lambda: True)
        for i, (x, y) in enumerate([(0.5, 0.5), (3.5, 0.5), (2, 2)])
    ]
    im = SimpleNamespace(
        points2D=points, camera_id=1, cam_from_world=lambda: pycolmap.Rigid3d()
    )
    model = SimpleNamespace(
        cameras={1: SimpleNamespace(width=4, height=4)},
        points3D={i: SimpleNamespace(xyz=np.array([1, 1, 0.5])) for i in range(3)},
    )
    np.testing.assert_allclose(chair.depth_errors(im, model, path, 2), [0, 0.5])


def test_pose_select_only_changes_pair_graph(monkeypatch, tmp_path):
    args = SimpleNamespace(
        dataset=tmp_path, output=tmp_path, threads=1, seed=0, cell_size=2
    )
    graphs = {"control": [("1", "2")], "pose": [("1", "3")]}
    calls = []

    def record(images, output, names, pairs, matcher, **kwargs):
        calls.append((pairs, kwargs))
        return {"best_model_id": 0, "models": [{}]}

    monkeypatch.setattr(chair, "reconstruct", record)
    priors = object()
    for arm in (
        "control_visual",
        "pose_visual",
        "pose_K",
        "pose_T",
        "pose_KT",
        "pose_KT_init",
    ):
        chair.run_arm(arm, args, ["1", "2", "3"], priors, graphs, lambda a, b: None)
    assert calls[0][0] != calls[1][0]
    assert calls[0][1] == calls[1][1]
    assert calls[1][1]["priors"] is None
    assert [
        (c["use_intrinsics_prior"], c["use_pose_prior"], c["init_from_poses"])
        for _, c in calls[2:]
    ] == [
        (True, False, False),
        (False, True, False),
        (True, True, False),
        (True, True, True),
    ]


def test_alignment_recovers_visual_scale():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    images = {
        i: SimpleNamespace(num_points3D=6, projection_center=lambda point=p: point)
        for i, p in enumerate(points)
    }
    ref = {}
    for i, pt in enumerate(points):
        ref[i] = np.eye(4)
        ref[i][:3, 3] = 2 * pt + [1.0, 2.0, 3.0]
    sim = chair.visual_alignment(images, ref, 0.02, 0)
    np.testing.assert_allclose(sim * points, 2 * points + [1, 2, 3], atol=1e-9)
