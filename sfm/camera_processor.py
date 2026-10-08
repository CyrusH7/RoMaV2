"""Convert Stray Scanner camera_matrix.csv / odometry.csv into intrinsics.yaml / camera_poses.yaml.

Usage:
    python sfm/camera_processor.py \
        --camera-matrix local/example/sofa/camera_matrix.csv \
        --odometry local/example/sofa/odometry.csv \
        --images-dir local/example/images \
        --output-dir local/example

Outputs (written to --output-dir):
    intrinsics.yaml    {camera_id: {params: [fx, fy, cx, cy], images: [<image file name>]}}
    camera_poses.yaml  {camera_poses: {<images-dir name>/<frame>: {transform_matrix: 4x4 cam_from_world}}}

Stray Scanner odometry stores the camera pose in the world frame (camera_to_world, ARKit convention).
The poses are inverted here so that they are COLMAP-style cam_from_world transforms.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger("camera_processor")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def read_camera_matrix(path: Path) -> np.ndarray:
    """Read the 3x3 intrinsic matrix."""
    with Path(path).open("r", encoding="utf-8") as f:
        rows = [[float(v) for v in row] for row in csv.reader(f) if row]
    matrix = np.array(rows, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"{path}: expected a 3x3 camera matrix, got {matrix.shape}")
    return matrix


def read_odometry(path: Path) -> list[dict]:
    """Read odometry rows: timestamp, frame, x, y, z, qx, qy, qz, qw."""
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        rows = [
            {
                "frame": row["frame"].strip(),
                **{k: float(row[k]) for k in ("x", "y", "z", "qx", "qy", "qz", "qw")},
            }
            for row in reader
        ]
    if not rows:
        raise ValueError(f"{path}: no odometry rows")
    return rows


def quaternion_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0:
        raise ValueError("zero-norm quaternion")
    qx, qy, qz, qw = quat / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def cam_from_world(odom: dict) -> list[list[float]]:
    world_from_cam = quaternion_to_rotation(odom["qx"], odom["qy"], odom["qz"], odom["qw"])
    center = np.array([odom["x"], odom["y"], odom["z"]], dtype=np.float64)
    transform = np.eye(4)
    transform[:3, :3] = world_from_cam.T
    transform[:3, 3] = -world_from_cam.T @ center
    return transform.tolist()


def index_images(images_dir: Path) -> dict[str, str]:
    """Map image stem -> file name."""
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images dir not found: {images_dir}")
    return {
        p.stem: p.name
        for p in sorted(images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }


def build_payloads(camera_matrix: np.ndarray, odometry: list[dict], images_dir: Path):
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    image_files = index_images(images_dir)

    poses, intrinsics = {}, {}
    missing = []
    for camera_id, odom in enumerate(odometry, start=1):
        frame = odom["frame"]
        if frame not in image_files:
            missing.append(frame)
            continue
        poses[f"{images_dir.name}/{frame}"] = {"transform_matrix": cam_from_world(odom)}
        intrinsics[camera_id] = {"params": [fx, fy, cx, cy], "images": [image_files[frame]]}

    if missing:
        logger.warning(f"{len(missing)} odometry frames have no image in {images_dir}, e.g. {missing[:3]}")
    if not poses:
        raise SystemExit(f"no odometry frame matches an image in {images_dir}")
    unposed = sorted(set(image_files) - {o["frame"] for o in odometry})
    if unposed:
        logger.warning(f"{len(unposed)} images have no odometry row, e.g. {unposed[:3]}")
    return {"camera_poses": poses}, intrinsics


def save_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera-matrix", type=Path, required=True, help="camera_matrix.csv (3x3)")
    parser.add_argument("--odometry", type=Path, required=True, help="odometry.csv")
    parser.add_argument("--images-dir", type=Path, required=True, help="directory with the RGB frames")
    parser.add_argument("--output-dir", type=Path, required=True, help="where intrinsics.yaml / camera_poses.yaml go")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(name)s %(levelname)s] %(message)s")
    camera_poses, intrinsics = build_payloads(
        read_camera_matrix(args.camera_matrix), read_odometry(args.odometry), args.images_dir
    )
    save_yaml(camera_poses, args.output_dir / "camera_poses.yaml")
    save_yaml(intrinsics, args.output_dir / "intrinsics.yaml")
    logger.info(f"{len(intrinsics)} cameras -> {args.output_dir}/{{intrinsics,camera_poses}}.yaml")


if __name__ == "__main__":
    main()
