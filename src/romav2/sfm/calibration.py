"""Explicit calibration records; scene defaults never leak into general SfM."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OFFICE_CALIBRATION = Path(__file__).with_name("configs") / "office.json"


@dataclass(frozen=True)
class Calibration:
    width: int
    height: int
    params: tuple[float, float, float, float]
    source: str
    model: str = "PINHOLE"

    def validate_size(self, size):
        if tuple(size) != (self.width, self.height):
            raise ValueError(
                f"Calibration is for {self.width}x{self.height}, image is {size}; "
                "rescale the intrinsics explicitly when resizing images"
            )

    def as_dict(self):
        from dataclasses import asdict

        return asdict(self)


def validate_camera_to_world(matrix, name):
    t = np.asarray(matrix, dtype=float)
    if (
        t.shape != (4, 4)
        or not np.isfinite(t).all()
        or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-4)
        or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-4)
    ):
        raise ValueError(f"Invalid rigid camera-to-world transform for {name}")
    return t


def validate_pinhole(params):
    values = np.asarray(params, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all() or np.any(values[:2] <= 0):
        raise ValueError("PINHOLE needs finite fx fy cx cy and positive focal lengths")
    return values


def load_calibration(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return calibration_from_dict(data, str(Path(path).resolve()))


def calibration_from_dict(data, fallback_source="unspecified"):
    if data.get("model") != "PINHOLE":
        raise ValueError("Calibration files currently support PINHOLE only")
    width, height = data.get("width"), data.get("height")
    if (
        not isinstance(width, int)
        or not isinstance(height, int)
        or isinstance(width, bool)
        or isinstance(height, bool)
        or min(width, height) <= 0
    ):
        raise ValueError("Calibration width and height must be positive integers")
    return Calibration(
        width,
        height,
        tuple(validate_pinhole(data["params"])),
        data.get("source", fallback_source),
    )
