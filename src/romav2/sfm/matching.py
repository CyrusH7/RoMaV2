"""Turn pairwise RoMa correspondences into shared per-image observations."""
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


class FeatureStore:
    """Merge observations in fixed pixel cells to make multi-view tracks possible.

    COLMAP pixel coordinates have the top-left pixel center at (0.5, 0.5).
    All pairwise observations in the same cell share a representative cell center.
    """
    def __init__(self, cell_size: float):
        if not np.isfinite(cell_size) or cell_size <= 0:
            raise ValueError('cell_size must be positive and finite')
        self.cell_size = cell_size
        self.cells = defaultdict(dict)
        self.points = defaultdict(list)
        self.matches = {}
        self.scores = {}

    def _indices(self, name, xy, size):
        cells = np.floor(xy / self.cell_size).astype(np.int64)
        indices = []
        for cell in cells:
            key = tuple(cell)
            if key not in self.cells[name]:
                self.cells[name][key] = len(self.points[name])
                self.points[name].append(np.minimum((cell + 0.5) * self.cell_size, np.asarray(size) - 0.5))
            indices.append(self.cells[name][key])
        return np.asarray(indices, dtype=np.uint32)

    def add(self, name_a, name_b, xy_a, xy_b, scores, sizes):
        a = np.asarray(xy_a, dtype=np.float64).reshape(-1, 2)
        b = np.asarray(xy_b, dtype=np.float64).reshape(-1, 2)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(a) != len(b) or len(a) != len(scores):
            raise ValueError('Correspondences and scores must have equal lengths')
        if (name_a, name_b) in self.matches:
            raise ValueError('Duplicate image pair')
        wa, ha = sizes[name_a]; wb, hb = sizes[name_b]
        valid = (np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1) & np.isfinite(scores)
                 & (a[:, 0] >= 0) & (a[:, 0] < wa) & (a[:, 1] >= 0) & (a[:, 1] < ha)
                 & (b[:, 0] >= 0) & (b[:, 0] < wb) & (b[:, 1] >= 0) & (b[:, 1] < hb))
        a, b, scores = a[valid], b[valid], scores[valid]
        # Keep at most one match per cell on either side, preferring confidence.
        a_cells = np.floor(a / self.cell_size).astype(np.int64)
        b_cells = np.floor(b / self.cell_size).astype(np.int64)
        used_a, used_b, selected = set(), set(), []
        for i in np.argsort(-scores, kind='stable'):
            ka, kb = tuple(a_cells[i]), tuple(b_cells[i])
            if ka not in used_a and kb not in used_b:
                used_a.add(ka); used_b.add(kb); selected.append(i)
        selected = np.asarray(selected, dtype=np.int64)
        ia = self._indices(name_a, a[selected], sizes[name_a]); ib = self._indices(name_b, b[selected], sizes[name_b])
        self.matches[name_a, name_b] = np.column_stack((ia, ib)).astype(np.uint32)
        self.scores[name_a, name_b] = scores[selected].astype(np.float32)

    def keypoints(self, name):
        return np.asarray(self.points[name], dtype=np.float32).reshape(-1, 2)


class RomaMatcher:
    def __init__(self, setting, max_matches, seed):
        import torch
        from romav2 import RoMaV2
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.model = RoMaV2()
        self.model.apply_setting(setting)
        self.max_matches = max_matches

    def __call__(self, a: Path, b: Path):
        with Image.open(a) as image:
            wa, ha = image.size
        with Image.open(b) as image:
            wb, hb = image.size
        predictions = self.model.match(a, b)
        matches, scores, _, _ = self.model.sample(predictions, self.max_matches)
        xy_a, xy_b = self.model.to_pixel_coordinates(matches, ha, wa, hb, wb)
        return xy_a.cpu().numpy(), xy_b.cpu().numpy(), scores.cpu().numpy()
