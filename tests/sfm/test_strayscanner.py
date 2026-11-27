from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from romav2.sfm.io import write_json
from romav2.sfm.priors import load_priors
from romav2.sfm.strayscanner import scanner_priors


def scanner_export(root: Path, per_frame=False):
    (root / "images").mkdir()
    for name in ("000000.png", "000007.png"):
        Image.new("RGB", (640, 480)).save(root / "images" / name)
    suffixes = (",fx,fy,cx,cy", ",501,502,320,240", ",505,506,321,239")
    header, first, second = suffixes if per_frame else ("", "", "")
    (root / "odometry.csv").write_text(
        f"timestamp, frame, x, y, z, qx, qy, qz, qw{header}\n"
        f"1.0,000007,1.5,0.5,-0.25,0,0,0.6,0.8{first}\n"
        f"0.0,000000,1,0,0,0,0,0,1{second}\n"
    )
    if not per_frame:
        (root / "camera_matrix.csv").write_text("500,0,320\n0,500,240\n0,0,1\n")


@pytest.mark.parametrize("per_frame", [False, True])
def test_scanner_pose_basis_and_calibration(tmp_path, per_frame):
    scanner_export(tmp_path, per_frame)
    path = tmp_path / "priors.json"
    doc = scanner_priors(tmp_path)
    write_json(path, doc)
    a, b = "000000.png", "000007.png"
    priors = load_priors(path, [a, b])

    np.testing.assert_allclose(priors.poses[a][:3, :3], np.eye(3))
    np.testing.assert_allclose(
        priors.poses[b][:3],
        [[0.28, -0.96, 0, 1.5], [0.96, 0.28, 0, 0.5], [0, 0, 1, -0.25]],
    )
    np.testing.assert_allclose(
        priors.cam_from_world(b) * np.array([1.5, 0.5, -0.25]), [0, 0, 0], atol=1e-12
    )
    np.testing.assert_allclose(
        priors.cameras[b], [501, 502, 320, 240] if per_frame else [500, 500, 320, 240]
    )
    assert (priors.calibration is None) == per_frame


def test_scanner_rejects_unmatched_image(tmp_path):
    scanner_export(tmp_path)
    (tmp_path / "images/000007.png").rename(tmp_path / "images/000008.png")
    with pytest.raises(ValueError, match="Missing odometry for image 000008.png"):
        scanner_priors(tmp_path)
