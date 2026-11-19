"""Regression checks for COLMAP boundaries and experiment provenance."""

import io
import json
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pycolmap
import pytest
from PIL import Image

from romav2.sfm.ablation import common_pose_metrics, prepare, read_depth_from_rgb
from romav2.sfm.calibration import OFFICE_CALIBRATION, load_calibration
from romav2.sfm.colmap import reconstruct
from romav2.sfm.matching import FeatureStore
from romav2.sfm.options import PipelineOptions


def test_office_calibration_and_size_guard(tmp_path):
    calibration = load_calibration(OFFICE_CALIBRATION)
    assert calibration.params == (532.57, 531.54, 320.0, 240.0)
    assert calibration.source == "user_provided"
    calibration.validate_size((640, 480))
    root = tmp_path / "images"
    root.mkdir()
    for name in ["a.png", "b.png"]:
        Image.new("RGB", (320, 240)).save(root / name)

    def forbidden(*args):
        raise AssertionError("Invalid calibration should fail before matching")

    with pytest.raises(ValueError, match="rescale"):
        reconstruct(
            root,
            tmp_path / "out",
            ["a.png", "b.png"],
            [("a.png", "b.png")],
            forbidden,
            calibration=calibration,
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "options",
    [
        dict(cell_size=float("nan")),
        dict(min_match_score=-0.1),
        dict(geometric_max_trials=1),
        dict(geometric_max_error=0),
        dict(num_threads=0),
    ],
)
def test_invalid_options_fail_preflight(options):
    with pytest.raises(ValueError):
        PipelineOptions(**options)


def test_colmap_coordinates_pair_dedup_and_failure_record(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    for name in ["a.png", "b.png"]:
        Image.new("RGB", (64, 64)).save(root / name)
    calls = []

    def matcher(a, b):
        calls.append((a.name, b.name))
        return (
            np.array([[1.25, 2.75], [20, 20]]),
            np.array([[5.1, 6.2], [30, 30]]),
            np.array([0.9, 0.05]),
        )

    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="did not reconstruct"):
        reconstruct(
            root,
            output,
            ["b.png", "a.png"],
            [("b.png", "a.png"), ("a.png", "b.png")],
            matcher,
            min_match_score=0.1,
            num_threads=1,
        )
    assert calls == [("a.png", "b.png")]
    with pycolmap.Database.open(str(output / "database.db")) as db:
        image = db.read_image_with_name("a.png")
        # Exactly one COLMAP-origin conversion; no extra +0.5 or cell snapping.
        assert np.allclose(db.read_keypoints(image.image_id), [[1.25, 2.75]])
    state = json.loads((output / "run.json").read_text())
    assert state["status"] == "failed"
    assert state["stage"] == "model_finalization"
    assert state["error"]["type"] == "RuntimeError"
    stats = json.loads((output / "pair_statistics.json").read_text())
    assert stats["a.png b.png"]["raw_matches"] == 1
    assert json.loads((output / "summary.json").read_text())["best_model_id"] is None


def test_mutual_best_matches_and_subpixel_representatives():
    store = FeatureStore(2)
    sizes = {"a": (64, 64), "b": (64, 64)}
    store.add(
        "a",
        "b",
        [[1.1, 1.2], [1.3, 1.4], [10.2, 10.3]],
        [[2.1, 2.2], [20.1, 20.2], [2.3, 2.4]],
        [0.9, 0.8, 0.7],
        sizes,
    )
    # Only the mutually strongest edge survives; neither a weaker alternative
    # nor a snapped grid center should be used as the feature coordinate.
    assert store.matches["a", "b"].shape == (1, 2)
    assert np.allclose(store.keypoints("a"), [[1.1, 1.2]])


def test_cache_corruption_is_recomputed(tmp_path, monkeypatch):
    import torch

    import romav2
    from romav2.sfm.matching import RomaMatcher

    class FakeModel:
        def apply_setting(self, setting):
            pass

        def match(self, a, b):
            return {}

        def sample(self, p, count):
            return torch.zeros(1, 4), torch.ones(1), None, None

        def to_pixel_coordinates(self, m, *sizes):
            return torch.ones(1, 2), torch.ones(1, 2)

    monkeypatch.setitem(romav2.__dict__, "RoMaV2", FakeModel)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    Image.new("RGB", (10, 10)).save(a)
    Image.new("RGB", (10, 10), (1, 2, 3)).save(b)
    cache = tmp_path / "cache"
    first = RomaMatcher("fast", 10, 0, cache)
    first(a, b)
    path = next(cache.glob("*.npz"))
    path.write_bytes(b"corrupt cache")
    second = RomaMatcher("fast", 10, 0, cache)
    result = second(a, b)
    assert second.statistics["invalid_cache_entries"] == 1
    assert second.statistics["cache_misses"] == 1
    assert result[0].shape == (1, 2)
    third = RomaMatcher("fast", 10, 0, cache)
    third(a, b)
    assert third.statistics["cache_hits"] == 1


def test_prepare_records_supplied_calibration(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    with ZipFile(dataset / "seq-01.zip", "w") as z:
        for i in range(3):
            prefix = f"seq-01/frame-{i:06}"
            image = io.BytesIO()
            Image.new("RGB", (640, 480)).save(image, format="PNG")
            z.writestr(prefix + ".color.png", image.getvalue())
            pose = np.eye(4)
            pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
            pose[:3, 3] = [i * 0.1, (i % 2) * 0.1, 0]
            text = io.StringIO()
            np.savetxt(text, pose)
            z.writestr(prefix + ".pose.txt", text.getvalue())
    output = tmp_path / "experiment"
    args = SimpleNamespace(
        dataset=dataset,
        sequence="seq-01",
        output=output,
        count=3,
        window=1,
        noise_m=0,
        noise_deg=0,
        position_sigma=0.05,
        rotation_sigma_deg=3,
        seed=0,
        calibration=OFFICE_CALIBRATION,
        intrinsics=None,
    )
    prepare(args)
    doc = json.loads((output / "priors.json").read_text())
    assert all(
        e["intrinsics"] == [532.57, 531.54, 320.0, 240.0]
        for e in doc["images"].values()
    )
    assert doc["calibration"]["source"] == "user_provided"
    reference = json.loads((output / "reference.json").read_text())
    assert reference["pose_sensor"] == doc["pose_sensor"] == "rgb"
    assert reference["source_pose_sensor"] == "depth"
    # The offset is expressed in the rotating depth camera, not the world.
    np.testing.assert_allclose(
        reference["images"]["frame-000001.color.png"]["camera_to_world"],
        [
            [0, -1, 0, 0.1 - 0.00617665950],
            [1, 0, 0, 0.1 + 0.0234490289],
            [0, 0, 1, 0.0105253125],
            [0, 0, 0, 1],
        ],
    )
    for name, entry in reference["images"].items():
        np.testing.assert_allclose(
            entry["camera_to_world"], doc["images"][name]["camera_to_world"]
        )
    assert json.loads((output / "run.json").read_text())["status"] == "complete"


def test_rig_calibration_rejects_reflections_and_wrong_units(tmp_path):
    path = tmp_path / "rig.json"
    for transform, units in [(np.diag([-1, 1, 1, 1]), "meters"), (np.eye(4), "mm")]:
        path.write_text(
            json.dumps({"units": units, "depth_from_rgb": transform.tolist()})
        )
        with pytest.raises(ValueError, match="depth_from_rgb in meters"):
            read_depth_from_rgb(path)


def test_common_subset_metrics_use_identical_frames():
    reference = {}
    for i, c in enumerate([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]]):
        pose = np.eye(4)
        pose[:3, 3] = c
        reference[str(i)] = pose
    a = {n: p.copy() for n, p in reference.items()}
    a["3"][:3, 3] += [10, 0, 0]  # Only baseline contains this high-error camera.
    b = {n: p.copy() for n, p in reference.items() if n != "3"}
    names, metrics = common_pose_metrics(
        {"baseline": a, "both": b},
        reference,
        {"baseline": {"raw_ate_rmse_m": None}, "both": {"raw_ate_rmse_m": 0}},
    )
    assert names == ["0", "1", "2"]
    assert (
        metrics["baseline"]["evaluated_images"]
        == metrics["both"]["evaluated_images"]
        == 3
    )
    assert metrics["baseline"]["sim3_ate_rmse_m"] < 1e-9
    assert metrics["baseline"]["raw_ate_rmse_m"] is None


def test_match_cache_can_live_inside_run_directory(tmp_path, monkeypatch):
    import torch

    import romav2
    from romav2.sfm.matching import RomaMatcher

    class FakeModel:
        def apply_setting(self, setting):
            pass

        def match(self, a, b):
            return {}

        def sample(self, predictions, count):
            return torch.zeros(1, 4), torch.ones(1), None, None

        def to_pixel_coordinates(self, matches, *sizes):
            return torch.ones(1, 2), torch.ones(1, 2)

    monkeypatch.setitem(romav2.__dict__, "RoMaV2", FakeModel)
    root = tmp_path / "images"
    root.mkdir()
    for name in ["a.png", "b.png"]:
        Image.new("RGB", (20, 20)).save(root / name)
    output = tmp_path / "out"
    matcher = RomaMatcher("fast", 1, 0, output / "match-cache")
    assert not output.exists()
    with pytest.raises(RuntimeError, match="did not reconstruct"):
        reconstruct(
            root,
            output,
            ["a.png", "b.png"],
            [("a.png", "b.png")],
            matcher,
            num_threads=1,
        )
    assert list((output / "match-cache").glob("*.npz"))
    assert json.loads((output / "run.json").read_text())["status"] == "failed"
