"""Image discovery and pair sources for the SfM pipeline."""

from itertools import combinations
from pathlib import Path

import numpy as np

from .calibration import validate_camera_to_world

DEFAULT_POSE_TOP_K = 20
ROTATION_POOL_MULTIPLIER = 3

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def discover_images(root: Path, pattern: str = "**/*") -> list[str]:
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}")
    names = sorted(
        p.relative_to(root).as_posix()
        for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(names) < 2:
        raise ValueError(f"At least two images are required in {root}")
    return names


def read_pairs(path: Path, names: list[str]) -> list[tuple[str, str]]:
    known, seen, pairs = set(names), set(), []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if (
            len(parts) != 2
            or any(p not in known for p in parts)
            or parts[0] == parts[1]
        ):
            raise ValueError(f"Invalid pair at {path}:{line_number}: {line!r}")
        pair = tuple(sorted(parts))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    if not pairs:
        raise ValueError(f"No valid pairs in {path}")
    return pairs


def exhaustive_pairs(names: list[str]) -> list[tuple[str, str]]:
    return list(combinations(names, 2))


def validate_pose_top_k(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError("Pose selection requires a positive integer top_k")
    return int(value)


def pairs_from_poses(names, poses, top_k=DEFAULT_POSE_TOP_K):
    """Select nearby centers from the 3K closest orientations per image."""
    top_k = validate_pose_top_k(top_k)
    names = sorted(names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError(
            "Pose selection requires at least two distinct image names and no duplicates"
        )
    missing = [name for name in names if name not in poses]
    if missing:
        raise ValueError(f"Missing camera-to-world pose for {missing[0]}")

    transforms = np.stack(
        [validate_camera_to_world(poses[name], name) for name in names]
    )
    centers = transforms[:, :3, 3]
    u, _, vt = np.linalg.svd(transforms[:, :3, :3])
    rotations = u @ vt
    pool_size = min(len(names) - 1, ROTATION_POOL_MULTIPLIER * top_k)
    pairs = []
    for i, name in enumerate(names):
        traces = np.einsum("nij,ij->n", rotations, rotations[i])
        cos_angles = np.clip((traces - 1) / 2, -1, 1)
        cos_angles[i] = -np.inf
        rotation_pool = np.argsort(-cos_angles, kind="stable")[:pool_size]
        distances = np.linalg.norm(centers[rotation_pool] - centers[i], axis=1)
        closest = np.argsort(distances, kind="stable")[:top_k]
        neighbors = rotation_pool[closest]
        pairs.extend((name, names[j]) for j in neighbors)
    return pairs


def normalize_pairs(pairs, names):
    """Canonicalize pair orientation for the API as well as text input."""
    known, seen = set(names), set()
    for pair in pairs:
        if len(pair) != 2 or pair[0] == pair[1] or any(n not in known for n in pair):
            raise ValueError(f"Invalid image pair: {pair!r}")
        seen.add(tuple(sorted(pair)))
    if not seen:
        raise ValueError("At least one valid image pair is required")
    return sorted(seen)
