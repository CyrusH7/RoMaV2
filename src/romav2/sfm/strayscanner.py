"""Convert Stray Scanner odometry and native RGB calibration to SfM priors."""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .calibration import Calibration, validate_pinhole
from .io import write_json
from .pairs import discover_images

POSE_COLUMNS = ("x", "y", "z", "qx", "qy", "qz", "qw")
INTRINSICS_COLUMNS = ("fx", "fy", "cx", "cy")


def _read_odometry(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, skipinitialspace=True)
        headers = set(reader.fieldnames or ())
        if not {"frame", *POSE_COLUMNS} <= headers:
            raise ValueError(f"Missing frame or pose columns in {path}")
        intrinsics_columns = headers.intersection(INTRINSICS_COLUMNS)
        if intrinsics_columns and len(intrinsics_columns) != len(INTRINSICS_COLUMNS):
            raise ValueError(f"Incomplete per-frame intrinsics columns in {path}")
        frames = {}
        for row in reader:
            try:
                frame_id = int(row["frame"])
            except (ValueError, TypeError) as error:
                raise ValueError(
                    f"Invalid frame ID at {path}:{reader.line_num}"
                ) from error
            if frame_id < 0 or frame_id in frames:
                raise ValueError(f"Invalid or duplicate frame ID {frame_id} in {path}")
            frames[frame_id] = row
    return frames, bool(intrinsics_columns)


def _shared_intrinsics(path):
    matrix = np.loadtxt(path, delimiter=",")
    if (
        matrix.shape != (3, 3)
        or not np.allclose(matrix[2], [0, 0, 1])
        or not np.allclose([matrix[0, 1], matrix[1, 0]], 0)
    ):
        raise ValueError(f"Expected a 3x3 PINHOLE matrix in {path}")
    fx, fy = matrix[0, 0], matrix[1, 1]
    cx, cy = matrix[:2, 2]
    return validate_pinhole([fx, fy, cx, cy]).tolist()


def scanner_priors(dataset):
    """Read native RGB frames from images/ and match them by odometry frame ID."""
    dataset = Path(dataset).resolve()
    image_root = dataset / "images"
    names = discover_images(image_root)
    odometry = dataset / "odometry.csv"
    frames, per_frame = _read_odometry(odometry)
    intrinsics_source = odometry if per_frame else dataset / "camera_matrix.csv"
    shared = None if per_frame else _shared_intrinsics(intrinsics_source)

    entries = {}
    seen_ids = set()
    width, height = None, None
    for name in names:
        try:
            frame_id = int(Path(name).stem)
        except ValueError as error:
            raise ValueError(
                f"Expected a numeric frame ID in image name {name}"
            ) from error
        if frame_id in seen_ids:
            raise ValueError(f"Multiple images for frame ID {frame_id}")
        if frame_id not in frames:
            raise ValueError(f"Missing odometry for image {name}")
        seen_ids.add(frame_id)
        with Image.open(image_root / name) as image:
            if width is None:
                width, height = image.size
            elif image.size != (width, height):
                raise ValueError(f"Inconsistent RGB image dimensions for {name}")

        row = frames[frame_id]
        try:
            values = np.array([float(row[key]) for key in POSE_COLUMNS])
            if not np.isfinite(values).all() or not np.isclose(
                np.linalg.norm(values[3:]), 1, atol=1e-3
            ):
                raise ValueError("Expected a finite center and unit quaternion")
            pose = np.eye(4)
            # Stray Scanner already applies ARKit's 180-degree camera X rotation.
            pose[:3, :3] = Rotation.from_quat(values[3:]).as_matrix()
            pose[:3, 3] = values[:3]
            intrinsics = (
                validate_pinhole(
                    [float(row[key]) for key in INTRINSICS_COLUMNS]
                ).tolist()
                if per_frame
                else shared
            )
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid odometry for image {name}: {error}") from error
        entries[name] = {"camera_to_world": pose.tolist(), "intrinsics": intrinsics}

    doc = {
        "pose_convention": "camera_to_world_opencv",
        "pose_sensor": "rgb",
        "units": "meters",
        "odometry_source": str(odometry),
        "intrinsics_source": str(intrinsics_source),
        "image_size": [width, height],
        "images": entries,
    }
    if shared is not None:
        doc["calibration"] = Calibration(
            width, height, tuple(shared), str(intrinsics_source)
        ).as_dict()
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Stray Scanner export with extracted native RGB frames in images/",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New directory for priors.json"
    )
    args = parser.parse_args()
    doc = scanner_priors(args.dataset)
    args.output.mkdir(parents=True)
    path = args.output / "priors.json"
    write_json(path, doc)
    print(
        f"Wrote RGB intrinsics and poses for {len(doc['images'])} frames to {path.resolve()}"
    )


if __name__ == "__main__":
    main()
