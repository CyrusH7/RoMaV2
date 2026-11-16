"""Image discovery and pair sources for the SfM pipeline."""
from itertools import combinations
from pathlib import Path

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def discover_images(root: Path) -> list[str]:
    names = sorted(p.relative_to(root).as_posix() for p in root.rglob('*')
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if len(names) < 2:
        raise ValueError(f'At least two images are required in {root}')
    return names


def read_pairs(path: Path, names: list[str]) -> list[tuple[str, str]]:
    known, seen, pairs = set(names), set(), []
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        parts = line.split()
        if len(parts) != 2 or any(p not in known for p in parts) or parts[0] == parts[1]:
            raise ValueError(f'Invalid pair at {path}:{line_number}: {line!r}')
        pair = tuple(sorted(parts))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    if not pairs:
        raise ValueError(f'No valid pairs in {path}')
    return pairs


def exhaustive_pairs(names: list[str]) -> list[tuple[str, str]]:
    return list(combinations(names, 2))
