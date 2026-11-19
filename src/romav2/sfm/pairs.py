"""Image discovery and pair sources for the SfM pipeline."""

from itertools import combinations
from pathlib import Path

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
