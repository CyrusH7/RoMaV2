"""Turn pairwise RoMa correspondences into shared per-image observations."""

import hashlib
import json
import logging
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from PIL import Image

LOG = logging.getLogger(__name__)


class FeatureStore:
    """Merge observations in fixed pixel cells to make multi-view tracks possible.

    COLMAP pixel coordinates have the top-left pixel center at (0.5, 0.5).
    All pairwise observations in one cell share a confidence-weighted pixel
    coordinate, preserving subpixel localization instead of snapping to centers.
    """

    def __init__(self, cell_size: float, min_match_score: float = 0.0):
        if not np.isfinite(cell_size) or cell_size <= 0:
            raise ValueError("cell_size must be positive and finite")
        if not np.isfinite(min_match_score) or not 0 <= min_match_score <= 1:
            raise ValueError("min_match_score must be in [0, 1]")
        self.min_match_score = min_match_score
        self.cell_size = cell_size
        self.cells = defaultdict(dict)
        self.points = defaultdict(list)
        self.weights = defaultdict(list)
        self.matches = {}
        self.scores = {}

    def _indices(self, name, xy, scores):
        cells = np.floor(xy / self.cell_size).astype(np.int64)
        indices = []
        for cell, point, score in zip(cells, xy, scores):
            key = tuple(cell)
            if key not in self.cells[name]:
                self.cells[name][key] = len(self.points[name])
                self.points[name].append(point.copy())
                self.weights[name].append(float(score))
            else:
                index = self.cells[name][key]
                weight = self.weights[name][index]
                self.points[name][index] += (point - self.points[name][index]) * (
                    score / (weight + score)
                )
                self.weights[name][index] += float(score)
            indices.append(self.cells[name][key])
        return np.asarray(indices, dtype=np.uint32)

    def add(self, name_a, name_b, xy_a, xy_b, scores, sizes):
        a = np.asarray(xy_a, dtype=np.float64).reshape(-1, 2)
        b = np.asarray(xy_b, dtype=np.float64).reshape(-1, 2)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(a) != len(b) or len(a) != len(scores):
            raise ValueError("Correspondences and scores must have equal lengths")
        if (name_a, name_b) in self.matches:
            raise ValueError("Duplicate image pair")
        wa, ha = sizes[name_a]
        wb, hb = sizes[name_b]
        valid = (
            np.isfinite(a).all(axis=1)
            & np.isfinite(b).all(axis=1)
            & np.isfinite(scores)
            & (a[:, 0] >= 0)
            & (a[:, 0] < wa)
            & (a[:, 1] >= 0)
            & (a[:, 1] < ha)
            & (scores > self.min_match_score)
            & (scores <= 1)
            & (b[:, 0] >= 0)
            & (b[:, 0] < wb)
            & (b[:, 1] >= 0)
            & (b[:, 1] < hb)
        )
        a, b, scores = a[valid], b[valid], scores[valid]
        # Retain only matches that are best for BOTH endpoint cells, as in hloc.
        a_cells = np.floor(a / self.cell_size).astype(np.int64)
        b_cells = np.floor(b / self.cell_size).astype(np.int64)
        best_a, best_b = {}, {}
        for i in np.argsort(-scores, kind="stable"):
            best_a.setdefault(tuple(a_cells[i]), i)
            best_b.setdefault(tuple(b_cells[i]), i)
        selected = np.array(
            sorted(set(best_a.values()) & set(best_b.values())), dtype=np.int64
        )
        ia = self._indices(name_a, a[selected], scores[selected])
        ib = self._indices(name_b, b[selected], scores[selected])
        self.matches[name_a, name_b] = np.column_stack((ia, ib)).astype(np.uint32)
        self.scores[name_a, name_b] = scores[selected].astype(np.float32)

    def keypoints(self, name):
        return np.asarray(self.points[name], dtype=np.float32).reshape(-1, 2)


class RomaMatcher:
    def __init__(self, setting, max_matches, seed, cache_dir=None):
        self.setting, self.max_matches, self.seed = setting, max_matches, seed
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._digests = {}
        self.model = None
        core = Path(__file__).resolve().parents[1]
        self.code_digest = hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(core.glob("*.py")))
            + Path(__file__).read_bytes()
        ).hexdigest()
        self.statistics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "invalid_cache_entries": 0,
        }

    def _model(self):
        if self.model is not None:
            return self.model
        import torch

        from romav2 import RoMaV2

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.model = RoMaV2()
        self.model.apply_setting(self.setting)
        return self.model

    def _digest(self, path):
        path = Path(path).resolve()
        key = (str(path), path.stat().st_size, path.stat().st_mtime_ns)
        if key not in self._digests:
            self._digests[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        return self._digests[key]

    def __call__(self, a: Path, b: Path):
        spec = {
            "version": 2,
            "model": "romav2.0.1",
            "setting": self.setting,
            "max_matches": self.max_matches,
            "seed": self.seed,
            "code": self.code_digest,
            "a": self._digest(a),
            "b": self._digest(b),
        }
        key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        cached = self.cache_dir / f"{key}.npz" if self.cache_dir else None
        if cached and cached.exists():
            try:
                with np.load(cached, allow_pickle=False) as data:
                    if json.loads(str(data["metadata"].item())) != spec:
                        raise ValueError(
                            "cache metadata does not match requested inference"
                        )
                    result = self._validate_arrays(
                        data["xy_a"], data["xy_b"], data["scores"]
                    )
                self.statistics["cache_hits"] += 1
                return result
            except (OSError, ValueError, KeyError, EOFError, BadZipFile) as error:
                LOG.warning("Recomputing invalid cache %s: %s", cached, error)
                self.statistics["invalid_cache_entries"] += 1
        self.statistics["cache_misses"] += 1
        import torch

        model = self._model()
        pair_seed = (self.seed + int(key[:8], 16)) % (2**32)
        torch.manual_seed(pair_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(pair_seed)
        with Image.open(a) as image:
            wa, ha = image.size
        with Image.open(b) as image:
            wb, hb = image.size
        predictions = model.match(a, b)
        matches, scores, _, _ = model.sample(predictions, self.max_matches)
        xy_a, xy_b = model.to_pixel_coordinates(matches, ha, wa, hb, wb)
        result = self._validate_arrays(
            xy_a.cpu().numpy(), xy_b.cpu().numpy(), scores.cpu().numpy()
        )
        if cached:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=self.cache_dir, suffix=".npz", delete=False
                ) as temp:
                    temporary = Path(temp.name)
                    np.savez_compressed(
                        temp,
                        xy_a=result[0],
                        xy_b=result[1],
                        scores=result[2],
                        metadata=np.array(json.dumps(spec, sort_keys=True)),
                    )
                os.replace(temporary, cached)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return result

    @staticmethod
    def _validate_arrays(xy_a, xy_b, scores):
        a, b, s = (np.asarray(x, dtype=np.float32) for x in (xy_a, xy_b, scores))
        if (
            a.ndim != 2
            or a.shape[1:] != (2,)
            or b.shape != a.shape
            or s.shape != (len(a),)
        ):
            raise ValueError("Matches must have shapes (N,2), (N,2), (N,)")
        if not all(np.isfinite(x).all() for x in (a, b, s)) or np.any(
            (s < 0) | (s > 1)
        ):
            raise ValueError(
                "Match coordinates/scores must be finite and scores must be in [0,1]"
            )
        return a, b, s

    def describe(self):
        return {
            "type": type(self).__name__,
            "model": "romav2.0.1",
            "setting": self.setting,
            "max_matches": self.max_matches,
            "seed": self.seed,
            "code_digest": self.code_digest,
            "cache_version": 2,
            "cache_dir": str(self.cache_dir.resolve()) if self.cache_dir else None,
            "coordinates": "COLMAP pixels; top-left pixel center is (0.5, 0.5)",
        }

    def release_model(self):
        """Return inference memory before COLMAP starts building its graph/BA."""
        if self.model is None:
            return
        import gc

        import torch

        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
