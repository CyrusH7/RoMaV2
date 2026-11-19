from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from romav2.sfm.ablation import prepare
from romav2.sfm.pairs import normalize_pairs, pairs_from_poses


def pose(degrees, x):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("y", degrees, degrees=True).as_matrix()
    transform[0, 3] = x
    return transform


def test_rotation_pool_then_center_distance():
    # The nearest center is outside the rotation pool. Within the pool,
    # the nearest center has a larger relative angle than two alternatives.
    poses = {
        str(i): pose(angle, x)
        for i, (angle, x) in enumerate(zip([0, 7, 19, 36, 62], [0, 4.1, 2.2, 1, 0.01]))
    }
    originals = {n: t.copy() for n, t in poses.items()}
    names = list(poses)
    ranked = pairs_from_poses(names, poses, 1)
    assert ranked[0] == ("0", "3")
    assert len(ranked) == len(names)
    assert all(a != b for a, b in ranked)
    world = pose(65, 18)
    transformed = {n: world @ t for n, t in poses.items()}
    for t in transformed.values():
        t[:3, 3] *= 3
    assert pairs_from_poses(names, transformed, 1) == ranked
    for n in names:
        np.testing.assert_array_equal(poses[n], originals[n])


def test_ties_exclude_self_and_keep_relative_image_names():
    names = ["b/1.png", "a/1.png", "a/2.png"]
    poses = {n: np.eye(4) for n in names}
    ranked = pairs_from_poses(names, poses)
    assert ranked == [(a, b) for a in sorted(names) for b in sorted(names) if a != b]
    assert len(normalize_pairs(ranked, names)) == 3
    assert pairs_from_poses(names[::-1], poses, 1) == [
        ("a/1.png", "a/2.png"),
        ("a/2.png", "a/1.png"),
        ("b/1.png", "a/1.png"),
    ]


def test_pose_selection_validates_inputs():
    names = ["a", "b"]
    poses = {n: np.eye(4) for n in names}
    for k in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            pairs_from_poses(names, poses, k)
    with pytest.raises(ValueError, match="Missing.*b"):
        pairs_from_poses(names, {"a": poses["a"]})
    poses["b"][0, 0] = -1
    with pytest.raises(ValueError, match="rigid.*b"):
        pairs_from_poses(names, poses)


def test_prepare_rejects_invalid_k_before_creating_output(tmp_path):
    output = tmp_path / "office"
    with pytest.raises(ValueError, match="positive integer"):
        prepare(SimpleNamespace(count=100, pose_select=0, output=output))
    assert not output.exists()
