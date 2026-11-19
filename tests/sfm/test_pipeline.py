import json
import sqlite3
from itertools import combinations

import numpy as np
import pycolmap
import pytest
from PIL import Image

from romav2.sfm.colmap import reconstruct
from romav2.sfm.matching import FeatureStore
from romav2.sfm.pairs import normalize_pairs, pairs_from_poses, read_pairs
from romav2.sfm.priors import Priors


def test_pairs_dedup_and_validation(tmp_path):
    path = tmp_path / "pairs.txt"
    path.write_text("a.png b.png\nb.png a.png\n# comment\na.png c.png\n")
    assert read_pairs(path, ["a.png", "b.png", "c.png"]) == [
        ("a.png", "b.png"),
        ("a.png", "c.png"),
    ]
    path.write_text("a.png missing.png\n")
    with pytest.raises(ValueError, match="Invalid pair"):
        read_pairs(path, ["a.png", "b.png"])


def test_merge_observations_and_keep_best():
    store = FeatureStore(1.0)
    sizes = {"a": (100, 100), "b": (100, 100), "c": (100, 100)}
    store.add(
        "a",
        "b",
        [[10.1, 20.1], [10.2, 20.2], [-1, 5]],
        [[30.1, 40.1], [50.1, 60.1], [5, 5]],
        [0.5, 0.9, 1],
        sizes,
    )
    store.add("a", "c", [[10.8, 20.8]], [[40.1, 40.1]], [0.8], sizes)
    assert store.matches["a", "b"].tolist() == [[0, 0]]
    assert store.matches["a", "c"].tolist() == [[0, 0]]
    assert np.allclose(
        store.keypoints("a"),
        [[(10.2 * 0.9 + 10.8 * 0.8) / 1.7, (20.2 * 0.9 + 20.8 * 0.8) / 1.7]],
    )
    assert np.allclose(store.keypoints("b"), [[50.1, 60.1]])


@pytest.mark.parametrize("pose_select_k", [None, np.int64(3)])
def test_colmap_reconstructs_shared_tracks(tmp_path, pose_select_k):
    images = tmp_path / "images"
    images.mkdir()
    names = [f"{i:03}.png" for i in range(5)]
    for name in names:
        Image.new("RGB", (640, 480), (120, 130, 140)).save(images / name)
    rng = np.random.default_rng(1)
    xyz = np.column_stack(
        (
            rng.uniform(-1.3, 1.3, 450),
            rng.uniform(-1, 1, 450),
            rng.uniform(4.0, 7.0, 450),
        )
    )
    coords, poses = {}, {}
    for i, name in enumerate(names):
        pose = np.eye(4)
        pose[:3, 3] = [0.6 * (i - 2), 0.04 * (i % 2), 0]
        poses[name] = pose
        camera_xyz = xyz - pose[:3, 3]
        coords[name] = np.column_stack(
            (
                500 * camera_xyz[:, 0] / camera_xyz[:, 2] + 320,
                500 * camera_xyz[:, 1] / camera_xyz[:, 2] + 240,
            )
        )

    calls = []

    def matcher(a, b):
        calls.append((a.name, b.name))
        return coords[a.name], coords[b.name], np.ones(len(xyz))

    output = tmp_path / "sfm"
    result = reconstruct(
        images,
        output,
        names,
        list(combinations(names, 2)) if pose_select_k is None else None,
        matcher,
        priors=Priors({}, poses) if pose_select_k is not None else None,
        pose_select_k=pose_select_k,
        camera_params=[500, 500, 320, 240],
        fix_intrinsics=True,
    )
    if pose_select_k is not None:
        assert calls == normalize_pairs(
            pairs_from_poses(names, poses, pose_select_k), names
        )
        assert calls == read_pairs(output / "pairs.txt", names)
        assert json.loads((output / "config.json").read_text())["pose_select_k"] == 3
        with sqlite3.connect(output / "database.db") as db:
            assert db.execute("SELECT COUNT(*) FROM pose_priors").fetchone() == (0,)
    assert result["verified_pairs"] >= 6
    assert max(m["registered_images"] for m in result["models"]) == 5
    model = pycolmap.Reconstruction(str(output / "sparse" / "0"))
    assert model.num_points3D() > 100
    assert np.allclose(next(iter(model.cameras.values())).params, [500, 500, 320, 240])
    with pytest.raises(ValueError, match="empty"):
        reconstruct(images, output, names, list(combinations(names, 2)), matcher)


def test_roma_matcher_converts_each_image_size(tmp_path, monkeypatch):
    import torch

    import romav2
    from romav2.sfm.matching import RomaMatcher

    class FakeModel:
        def apply_setting(self, setting):
            assert setting == "fast"

        def match(self, a, b):
            return (a, b)

        def sample(self, predictions, count):
            assert count == 3
            return (
                torch.tensor([[0.0, 0.0, -0.5, 0.5]]),
                torch.tensor([0.8]),
                None,
                None,
            )

        def to_pixel_coordinates(self, matches, ha, wa, hb, wb):
            assert (ha, wa, hb, wb) == (100, 200, 60, 80)
            return (
                (matches[:, :2] + 1) * torch.tensor([wa, ha]) / 2,
                (matches[:, 2:] + 1) * torch.tensor([wb, hb]) / 2,
            )

    monkeypatch.setitem(romav2.__dict__, "RoMaV2", FakeModel)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    Image.new("RGB", (200, 100)).save(a)
    Image.new("RGB", (80, 60)).save(b)
    x, y, scores = RomaMatcher("fast", 3, 0)(a, b)
    assert np.allclose(x, [[100, 50]])
    assert np.allclose(y, [[20, 45]])
    assert np.allclose(scores, [0.8])


def test_empty_matches_reports_no_model(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for name in ("a.png", "b.png"):
        Image.new("RGB", (64, 64)).save(images / name)

    def empty_matcher(a, b):
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)

    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="did not reconstruct"):
        reconstruct(
            images, output, ["a.png", "b.png"], [("a.png", "b.png")], empty_matcher
        )
    assert (output / "summary.json").exists()


def test_conflicting_verified_tracks_keep_higher_confidence():
    from romav2.sfm.tracks import consistent_edge_mask

    # Two different observations from image 0 become connected via images 1 and 2.
    image_ids = np.array([0, 0, 1, 2])
    kept = consistent_edge_mask(image_ids, [0, 2, 3], [2, 3, 1], [0.9, 0.8, 0.7])
    assert kept.tolist() == [True, True, False]
    # The result is driven by confidence even when the input pair order differs.
    kept = consistent_edge_mask(image_ids, [3, 0, 2], [1, 2, 3], [0.7, 0.9, 0.8])
    assert kept.tolist() == [False, True, True]


def test_sampling_no_overlap_and_oversized_request_on_cpu():
    import torch

    from romav2.sampling import sample_predictions

    predictions = {
        "warp_AB": torch.full((1, 8, 8, 2), 2.0),
        "overlap_AB": torch.ones(1, 8, 8, 1),
        "precision_AB": torch.eye(2).expand(1, 8, 8, 2, 2),
    }
    matches, scores, _, _ = sample_predictions(predictions, 20, bidirectional=False)
    assert matches.shape == (0, 4) and scores.shape == (0,)
    predictions["warp_AB"].zero_()
    matches, scores, _, _ = sample_predictions(predictions, 100, bidirectional=False)
    assert len(matches) == len(scores) == 64
    assert torch.isfinite(matches).all()
    assert torch.isfinite(scores).all()


def test_cpu_sampling_ignores_zero_and_nonfinite_confidence():
    import torch

    from romav2.sampling import sample_predictions

    confidence = torch.zeros(1, 8, 8, 1)
    confidence[0, 1, 1] = 0.9
    confidence[0, 2, 2] = float("nan")
    predictions = {
        "warp_AB": torch.zeros(1, 8, 8, 2),
        "overlap_AB": confidence,
        "precision_AB": None,
    }
    matches, scores, _, _ = sample_predictions(predictions, 50, bidirectional=False)
    assert len(matches) == 1
    assert torch.allclose(scores, torch.tensor([0.9]))


def test_cpu_correlation_avoids_cuda_kernel(monkeypatch):
    import torch

    import romav2.local_correlation as correlation

    class FailIfCalled:
        def local_corr(self, *args, **kwargs):
            raise AssertionError("CUDA extension used for CPU tensors")

    monkeypatch.setattr(correlation, "local_corr", FailIfCalled())
    feature = torch.ones(1, 1, 2, 2)
    warp = torch.zeros(1, 2, 2, 2)
    result = correlation.local_correlation(feature, feature, 0, warp, torch.ones(2))
    assert result.shape == (1, 1, 2, 2)
    assert torch.allclose(result, torch.ones_like(result))
