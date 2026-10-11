"""Sampson-residual / Levenberg-Marquardt refinement of the prior (ARKit) poses before triangulation.

Triangulation with known poses is very sensitive to the quality of the prior poses: hloc's pose-based geometric
verification gates every match with an epipolar-distance test against the prior relative pose, and the
triangulated points inherit whatever error the poses carry. This module repairs the priors in two stages, using
only the dense matches (feats / matches h5 written by `reconstruct.py`) and the fixed intrinsics.

Stage 1, initial pair (relative pose, 6-DoF increment, epipolar geometry only)
    * pick the pair whose prior geometry already explains most matches and has enough parallax;
    * back-project the matches to normalised coordinates x = K^-1 [u, v, 1]^T and build the essential matrix
      E = [t]_x R from the prior relative pose (x1 = R x0 + t);
    * approximate the epipolar geometric error by the (differentiable) Sampson residual
          r = (x1^T E x0) / sqrt((E x0)_1^2 + (E x0)_2^2 + (E^T x1)_1^2 + (E^T x1)_2^2)      [x f -> px]
    * minimise the Huber cost of r with LM on a 6-D increment d = [omega, tau] (R <- exp(omega) R, t <- t + tau,
      Jacobians by forward-mode autodiff), re-selecting the inliers on a shrinking px threshold schedule;
    * the Sampson residual is invariant to the scale of E, so only the direction of t is observable: the
      baseline length is restored afterwards from the metric prior baseline.
    The first camera keeps its prior pose, the second one becomes `cam1_from_cam0(refined) * prior_pose_0`.

Stage 2, all other images (absolute poses from 2D-3D matches)
    * the inlier matches of the initial pair are triangulated (positive depth, parallax and reprojection checked);
      through the shared hloc keypoints they directly give 2D-3D correspondences for every image matched with
      the pair;
    * images are processed in order of the number of correspondences. The pose of an image is refined by LM on the
      Huber reprojection error, starting from its prior (LO-RANSAC PnP as a fallback), and only accepted if it
      explains at least as many correspondences as the prior pose;
    * each registered image then extends the map (new points from its matches with registered images, existing
      tracks extended), which supplies correspondences to the images that follow.
    Images that cannot be registered keep their prior pose.
    * optional (RefineConfig.local_ba_every / global_ba_every / final_ba, off by default): COLMAP/Ceres bundle
      adjustment inside the loop, a local BA of the last `local_ba_window` images every N registrations and a global
      BA at the end. On 7-Scenes it did NOT remove the rotation drift (see `sfm/refine_only_eval.py`).

The refined poses replace the priors in `reconstruct.py` for geometric verification, triangulation and as the
starting point of the global BA.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

logger = logging.getLogger("pose_refine")

DTYPE = torch.float64


@dataclass
class RefineConfig:
    inlier_px: float = 2.0  # final inlier threshold (Sampson / reprojection), same order as hloc's match quantisation
    huber_px: float = 1.0  # Huber scale of the LM cost
    coarse_px: tuple = (16.0, 8.0, 4.0)  # shrinking inlier thresholds that precede inlier_px
    max_iters: int = 50

    init_min_matches: int = 100
    init_min_baseline: float = 0.05  # m, prior baseline of an initial pair
    init_min_angle_deg: float = 3.0  # median triangulation angle (prior geometry) of an initial pair
    init_max_trials: int = 5  # candidate pairs tried, the one registering most images wins
    init_min_inliers: int = 50  # Sampson inliers needed after refinement

    align_gauge: bool = False  # express the result in the frame / metric scale of the prior (Sim(3) over all cameras)

    min_corr: int = 20  # 2D-3D correspondences needed to attempt an image
    min_inliers: int = 15  # inliers needed to accept a refined image pose
    new_point_min_angle_deg: float = 1.5  # parallax of newly triangulated map points
    new_point_max_reproj_px: float = 2.0

    # Bundle adjustment inside the incremental stage (all off by default = the original open-loop behaviour).
    # Without it every image is registered against a map that is never corrected, so rotation / position errors
    # accumulate along the registration order.
    local_ba_every: int = 0  # local BA after every N newly registered images (0 = off)
    local_ba_window: int = 12  # the most recently registered images whose poses (and their points) are free
    global_ba_every: int = 0  # global BA each time the number of registered images grew by N (0 = off)
    final_ba: bool = False  # global BA (2 rounds, outlier filtering in between) after the last image
    ba_huber_px: float = 1.0  # Huber scale of the BA reprojection loss
    ba_max_reproj_px: float = 3.0  # observations above this after a BA are removed from the map
    ba_max_points: int = 40000  # global BA: this many longest tracks are optimised
    ba_max_iters_local: int = 30
    ba_max_iters_global: int = 60


def skew(v: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros((), dtype=v.dtype)
    return torch.stack(
        [
            torch.stack([zero, -v[2], v[1]]),
            torch.stack([v[2], zero, -v[0]]),
            torch.stack([-v[1], v[0], zero]),
        ]
    )


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    th2 = (w * w).sum()
    small = th2 < 1e-12
    th2s = torch.where(small, torch.ones_like(th2), th2)
    th = th2s.sqrt()
    a = torch.where(small, 1.0 - th2 / 6.0, torch.sin(th) / th)
    b = torch.where(small, 0.5 - th2 / 24.0, (1.0 - torch.cos(th)) / th2s)
    k = skew(w)
    return torch.eye(3, dtype=w.dtype) + a * k + b * (k @ k)


def sampson_px(R: torch.Tensor, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, f: float) -> torch.Tensor:
    """Signed Sampson distance in pixels of normalised matches under x1 = R x0 + t (E = [t]_x R)."""
    e = skew(t) @ R
    h0 = torch.cat([x0, torch.ones_like(x0[:, :1])], dim=1)
    h1 = torch.cat([x1, torch.ones_like(x1[:, :1])], dim=1)
    ex0 = h0 @ e.T
    etx1 = h1 @ e
    num = (h1 * ex0).sum(dim=1)
    den = torch.sqrt(ex0[:, 0] ** 2 + ex0[:, 1] ** 2 + etx1[:, 0] ** 2 + etx1[:, 1] ** 2 + 1e-18)
    return f * num / den


def reprojection_px(R: torch.Tensor, t: torch.Tensor, xyz: torch.Tensor, obs: torch.Tensor, f: float) -> torch.Tensor:
    """Reprojection residual (N, 2) in pixels of world points seen at normalised coordinates `obs`."""
    cam = xyz @ R.T + t
    z = cam[:, 2:3].clamp(min=1e-6)
    return f * (cam[:, :2] / z - obs)


def huber_weights(norm: torch.Tensor, k: float) -> torch.Tensor:
    return torch.where(norm <= k, torch.ones_like(norm), k / norm.clamp(min=1e-12))


def huber_cost(norm: torch.Tensor, k: float) -> torch.Tensor:
    return torch.where(norm <= k, 0.5 * norm**2, k * (norm - 0.5 * k)).sum()


def robust_lm(residual_fn, R, t, *, huber, max_iters, unit_translation, rel_tol=1e-9):
    """Levenberg-Marquardt on the Huber cost over a 6-D increment [omega, tau] of the pose (R, t).

    `residual_fn(R, t)` returns (N, k) residuals in pixels. With `unit_translation` the translation is kept at
    unit length after every step (scale-free epipolar residual).
    """

    def compose(R0, t0, d):
        Rn = so3_exp(d[:3]) @ R0
        tn = t0 + d[3:]
        if unit_translation:
            tn = tn / tn.norm()
        return Rn, tn

    zero = torch.zeros(6, dtype=DTYPE)
    r = residual_fn(R, t)
    cost = huber_cost(r.norm(dim=1), huber)
    cost0, lam, iters = cost.item(), 1e-3, 0
    for iters in range(1, max_iters + 1):
        jac = torch.func.jacfwd(lambda d: residual_fn(*compose(R, t, d)))(zero)  # (N, k, 6)
        r = residual_fn(R, t)
        w = huber_weights(r.norm(dim=1), huber)
        jac = jac.reshape(-1, 6)
        sw = w.repeat_interleave(r.shape[1])
        hess = jac.T @ (jac * sw[:, None])
        grad = jac.T @ (sw * r.reshape(-1))
        diag = torch.diagonal(hess).clamp(min=1e-9)
        improved = False
        while lam < 1e10:
            step = torch.linalg.solve(hess + lam * torch.diag(diag), -grad)
            Rn, tn = compose(R, t, step)
            new_cost = huber_cost(residual_fn(Rn, tn).norm(dim=1), huber)
            if new_cost < cost:
                improved = True
                break
            lam *= 5.0
        if not improved:
            break
        converged = (cost - new_cost) < rel_tol * cost or step.norm() < 1e-10
        R, t, cost, lam = Rn, tn, new_cost, max(lam / 3.0, 1e-9)
        if converged:
            break
    return R, t, {"cost_init": cost0, "cost_final": float(cost), "iters": iters}


def triangulate_dlt(P0: np.ndarray, P1: np.ndarray, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Linear triangulation of normalised points; P are 3x4 [R|t]. Returns (N, 3), NaN where degenerate."""
    a = np.stack(
        [
            x0[:, 0:1] * P0[2] - P0[0],
            x0[:, 1:2] * P0[2] - P0[1],
            x1[:, 0:1] * P1[2] - P1[0],
            x1[:, 1:2] * P1[2] - P1[1],
        ],
        axis=1,
    )
    vt = np.linalg.svd(a)[2]
    h = vt[:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        xyz = h[:, :3] / h[:, 3:4]
    xyz[np.abs(h[:, 3]) < 1e-12] = np.nan
    return xyz


def project(R: np.ndarray, t: np.ndarray, xyz: np.ndarray):
    cam = xyz @ R.T + t
    with np.errstate(divide="ignore", invalid="ignore"):
        return cam[:, :2] / cam[:, 2:3], cam[:, 2]


def triangulation_angle_deg(c0: np.ndarray, c1: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    d0, d1 = xyz - c0, xyz - c1
    cos = (d0 * d1).sum(1) / (np.linalg.norm(d0, axis=1) * np.linalg.norm(d1, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def relative_pose(Ra, ta, Rb, tb):
    """cam_b_from_cam_a for two cam_from_world poses."""
    R = Rb @ Ra.T
    return R, tb - R @ ta


def sampson_np(R: np.ndarray, t: np.ndarray, x0: np.ndarray, x1: np.ndarray, f: float) -> np.ndarray:
    with torch.no_grad():
        return sampson_px(
            torch.from_numpy(R), torch.from_numpy(t), torch.from_numpy(x0), torch.from_numpy(x1), f
        ).numpy()


class MatchGraph:
    """Keypoints (normalised) and index matches of all pairs, read from the hloc feats / matches files."""

    def __init__(self, feat_path: Path, match_path: Path, pairs_path: Path, names: list[str], intrinsics):
        fx, fy, cx, cy = intrinsics
        self.K = (float(fx), float(fy), float(cx), float(cy))
        self.f = 0.5 * (fx + fy)
        self.names = names
        self.xn: dict[str, np.ndarray] = {}
        with h5py.File(feat_path, "r") as hf:
            for name in names:
                kp = hf[name]["keypoints"].__array__().astype(np.float64) + 0.5  # COLMAP pixel origin
                self.xn[name] = np.stack([(kp[:, 0] - cx) / fx, (kp[:, 1] - cy) / fy], axis=1)
        self.pairs: dict[tuple[str, str], np.ndarray] = {}
        self.nbrs: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {n: [] for n in names}
        seen = set()
        with h5py.File(match_path, "r") as hm:
            for line in Path(pairs_path).read_text().splitlines():
                if not line.strip():
                    continue
                a, b = line.split()[:2]
                if (a, b) in seen or (b, a) in seen:
                    continue
                seen.add((a, b))
                m0 = self._read_matches(hm, a, b)
                if len(m0) == 0:
                    continue
                self.pairs[(a, b)] = m0
                self.nbrs[a].append((b, m0[:, 0], m0[:, 1]))
                self.nbrs[b].append((a, m0[:, 1], m0[:, 0]))

    @staticmethod
    def _read_matches(hm: h5py.File, a: str, b: str) -> np.ndarray:
        from hloc.utils.parsers import names_to_pair

        key, reverse = names_to_pair(a, b), False
        if key not in hm:
            key, reverse = names_to_pair(b, a), True
        m = hm[key]["matches0"].__array__()
        idx = np.flatnonzero(m != -1)
        out = np.stack([idx, m[idx]], axis=1).astype(np.int64)
        return out[:, ::-1].copy() if reverse else out


def pose_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = R, t
    return m


def epipolar_stats(graph: MatchGraph, poses: dict[str, np.ndarray], thresholds=(2.0, 4.0)) -> dict:
    """Epipolar consistency of all matches with a set of poses (Sampson px, per-pair inlier ratios)."""
    ratios = {th: [] for th in thresholds}
    medians, all_err = [], []
    for (a, b), m in graph.pairs.items():
        R, t = relative_pose(poses[a][:3, :3], poses[a][:3, 3], poses[b][:3, :3], poses[b][:3, 3])
        if np.linalg.norm(t) < 1e-9:
            continue
        err = np.abs(sampson_np(R, t, graph.xn[a][m[:, 0]], graph.xn[b][m[:, 1]], graph.f))
        medians.append(np.median(err))
        all_err.append(err)
        for th in thresholds:
            ratios[th].append(np.mean(err <= th))
    all_err = np.concatenate(all_err)
    out = {
        "sampson_median_px": float(np.median(all_err)),
        "sampson_mean_clipped_px": float(np.minimum(all_err, 8.0).mean()),
        "pair_median_of_medians_px": float(np.median(medians)),
    }
    for th in thresholds:
        out[f"inlier_ratio@{th:g}px"] = float(np.mean(ratios[th]))
    return out


class PoseRefiner:
    def __init__(self, graph: MatchGraph, names: list[str], poses: dict[str, np.ndarray], cfg: RefineConfig):
        self.g = graph
        self.names = names
        self.prior = {n: poses[n].copy() for n in names}
        for m in self.prior.values():  # GT-derived priors (7-Scenes) are only ~1e-3 orthonormal; keep R a proper rotation
            u, _, vt = np.linalg.svd(m[:3, :3])
            m[:3, :3] = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
        self.cfg = cfg
        self.f = graph.f

    def _thresholds(self):
        return (*self.cfg.coarse_px, self.cfg.inlier_px)

    def init_candidates(self) -> list[dict]:
        """Rank pairs by how many matches the prior geometry explains and by parallax, subject to baseline / parallax gates."""
        cfg, out = self.cfg, []
        for (a, b), m in self.g.pairs.items():
            if len(m) < cfg.init_min_matches:
                continue
            Ra, ta, Rb, tb = (*self._rt(a), *self._rt(b))
            R, t = relative_pose(Ra, ta, Rb, tb)
            baseline = float(np.linalg.norm(t))
            if baseline < cfg.init_min_baseline:
                continue
            x0, x1 = self.g.xn[a][m[:, 0]], self.g.xn[b][m[:, 1]]
            err = np.abs(sampson_np(R, t, x0, x1, self.f))
            ok = np.flatnonzero(err <= cfg.coarse_px[0])
            if len(ok) < cfg.init_min_matches:
                continue
            sub = ok[:: max(1, len(ok) // 300)]
            xyz = triangulate_dlt(proj_matrix(Ra, ta), proj_matrix(Rb, tb), x0[sub], x1[sub])
            good = np.isfinite(xyz).all(1)
            if good.sum() < 20:
                continue
            ca, cb = -Ra.T @ ta, -Rb.T @ tb
            angle = float(np.median(triangulation_angle_deg(ca, cb, xyz[good])))
            if angle < cfg.init_min_angle_deg:
                continue
            out.append({"pair": (a, b), "prior_inliers": int(len(ok)), "angle_deg": angle, "baseline": baseline})
        # many consistent matches, but a longer baseline pins down the scale / depth of the initial map
        out.sort(key=lambda c: -c["prior_inliers"] * min(c["angle_deg"], 10.0) / 10.0)
        return out

    def _rt(self, name):
        p = self.prior[name]
        return p[:3, :3], p[:3, 3]

    def refine_pair(self, a: str, b: str) -> dict | None:
        """Stage 1 on one pair; returns refined cam_b_from_cam_a and bookkeeping, or None."""
        cfg = self.cfg
        m = self.g.pairs[(a, b)]
        x0 = torch.from_numpy(self.g.xn[a][m[:, 0]])
        x1 = torch.from_numpy(self.g.xn[b][m[:, 1]])
        Rp, tp = relative_pose(*self._rt(a), *self._rt(b))
        baseline = float(np.linalg.norm(tp))
        R, t = torch.from_numpy(Rp), torch.from_numpy(tp / baseline)

        def residuals(sel):
            return lambda R_, t_: sampson_px(R_, t_, x0[sel], x1[sel], self.f)[:, None]

        with torch.no_grad():
            before = sampson_px(R, t, x0, x1, self.f).abs().numpy()
        info = {}
        for th in self._thresholds():
            with torch.no_grad():
                sel = torch.from_numpy(np.flatnonzero(sampson_px(R, t, x0, x1, self.f).abs().numpy() <= th))
            if len(sel) < 8:
                return None
            R, t, info = robust_lm(
                residuals(sel), R, t, huber=min(cfg.huber_px, th), max_iters=cfg.max_iters, unit_translation=True
            )
        with torch.no_grad():
            after = sampson_px(R, t, x0, x1, self.f).abs().numpy()
        inliers = after <= cfg.inlier_px
        if inliers.sum() < cfg.init_min_inliers:
            return None
        R, t = R.numpy(), t.numpy() * baseline  # restore the metric baseline of the prior
        return {
            "R": R,
            "t": t,
            "inlier_mask": inliers,
            "baseline": baseline,
            "stats": {
                "num_matches": int(len(m)),
                "num_inliers_before": int((before <= cfg.inlier_px).sum()),
                "num_inliers_after": int(inliers.sum()),
                "sampson_median_before_px": float(np.median(before)),
                "sampson_median_after_px": float(np.median(after)),
                "sampson_inlier_mean_before_px": float(before[before <= cfg.inlier_px].mean())
                if (before <= cfg.inlier_px).any()
                else None,
                "sampson_inlier_mean_after_px": float(after[inliers].mean()),
                "rotation_change_deg": rotation_angle_deg(R @ Rp.T),
                "translation_dir_change_deg": float(
                    np.degrees(np.arccos(np.clip(np.dot(t, tp) / (np.linalg.norm(t) * baseline), -1, 1)))
                ),
                "baseline_m": baseline,
                "lm": info,
            },
        }

    def run(self) -> tuple[dict[str, np.ndarray], dict]:
        t_start = time.time()
        candidates = self.init_candidates()
        logger.info(f"{len(candidates)} candidate initial pairs")
        best = None
        for cand in candidates[: self.cfg.init_max_trials]:
            a, b = cand["pair"]
            pair = self.refine_pair(a, b)
            if pair is None:
                logger.info(f"initial pair {cand['pair']}: refinement failed")
                continue
            result = self.register_all(a, b, pair)
            logger.info(
                f"initial pair ({a}, {b}): Sampson inliers {pair['stats']['num_inliers_before']} -> "
                f"{pair['stats']['num_inliers_after']}, registered {result['num_registered']}/{len(self.names)} images"
            )
            if best is None or result["num_registered"] > best["num_registered"]:
                best = {**result, "pair": (a, b), "init": pair["stats"], "candidate": cand}
            if result["num_registered"] == len(self.names):
                break
        if best is None:
            logger.warning("no usable initial pair, keeping the prior poses")
            return self.prior, {"success": False}
        stats = {
            "success": True,
            "initial_pair": list(best["pair"]),
            "initial_pair_prior_angle_deg": best["candidate"]["angle_deg"],
            "initial_pair_stage": best["init"],
            "num_registered": best["num_registered"],
            "num_images": len(self.names),
            "num_map_points": best["num_points"],
            "unregistered": best["unregistered"],
            "image_inliers": best["image_inliers"],
            "ba": best.get("ba", []),
            "elapsed_s": time.time() - t_start,
        }
        poses = best["poses"]
        if self.cfg.align_gauge:
            aligned_names = [n for n in self.names if n not in best["unregistered"]]
            aligned, scale = align_to_prior(poses, self.prior, aligned_names)
            poses = {n: aligned.get(n, self.prior[n]) for n in self.names}
            stats["gauge_scale_to_prior"] = scale
            logger.info(f"gauge alignment of the refined poses to the prior: scale {scale:.4f}")
        return poses, stats

    def register_all(self, a: str, b: str, pair: dict) -> dict:
        cfg, g = self.cfg, self.g
        Ra, ta = self._rt(a)
        Rab, tab = pair["R"], pair["t"]
        poses = {a: (Ra, ta), b: (Rab @ Ra, Rab @ ta + tab)}
        kp2pt = {n: np.full(len(g.xn[n]), -1, dtype=np.int64) for n in self.names}
        xyz = np.zeros((0, 3))

        def add_points(points: np.ndarray) -> np.ndarray:
            nonlocal xyz
            ids = np.arange(len(xyz), len(xyz) + len(points))
            xyz = np.concatenate([xyz, points])
            return ids

        m = g.pairs[(a, b)]
        sel = pair["inlier_mask"]
        self._extend_map(a, b, m[sel, 0], m[sel, 1], poses, kp2pt, add_points, lambda: xyz)
        if len(xyz) < cfg.init_min_inliers // 2:
            return {"num_registered": 2, "num_points": len(xyz), "poses": self.prior, "unregistered": [],
                    "image_inliers": {}}

        image_inliers: dict[str, int] = {}
        failed_at: dict[str, int] = {}
        reg_order = [a, b]
        since_ba, last_global, ba_log = 0, 2, []

        def run_ba(free, global_ba, tag):
            info = self._bundle_adjust(poses, kp2pt, xyz, anchors=(a, b), free=free, global_ba=global_ba)
            if info is not None:
                info["tag"], info["registered"] = tag, len(poses)
                ba_log.append(info)
                logger.info(
                    f"[{tag} BA @ {len(poses)} images] points {info['points']}, reproj {info['reproj_before']:.2f} -> "
                    f"{info['reproj_after']:.2f}px, moved rot mean {info['rot_move_deg']:.2f}deg / "
                    f"centre mean {info['pos_move_cm']:.1f}cm, {info['seconds']:.1f}s"
                )

        while True:
            best_name, best_corr = None, None
            for u in self.names:
                if u in poses:
                    continue
                corr = self._correspondences(u, poses, kp2pt)
                if len(corr[0]) < cfg.min_corr or len(corr[0]) <= failed_at.get(u, 0) * 1.2:
                    continue
                if best_corr is None or len(corr[0]) > len(best_corr[0]):
                    best_name, best_corr = u, corr
            if best_name is None:
                break
            u = best_name
            new = self._register_image(u, best_corr, xyz, poses)
            if new is None:
                failed_at[u] = len(best_corr[0])
                continue
            poses[u], image_inliers[u] = new[:2], new[2]
            for n, iu, in_ in g.nbrs[u]:
                if n in poses:
                    self._extend_map(u, n, iu, in_, poses, kp2pt, add_points, lambda: xyz)
            reg_order.append(u)
            since_ba += 1
            if cfg.global_ba_every and len(poses) - last_global >= cfg.global_ba_every:
                run_ba(None, True, "global")
                last_global, since_ba = len(poses), 0
            elif cfg.local_ba_every and since_ba >= cfg.local_ba_every:
                run_ba(reg_order[-cfg.local_ba_window:], False, "local")
                since_ba = 0
        if cfg.final_ba and len(poses) > 2:
            for round_idx in range(2):
                run_ba(None, True, f"final{round_idx + 1}")
        refined = dict(self.prior)
        for n, (R, t) in poses.items():
            refined[n] = pose_matrix(R, t)
        return {
            "num_registered": len(poses),
            "num_points": len(xyz),
            "poses": refined,
            "unregistered": [n for n in self.names if n not in poses],
            "image_inliers": image_inliers,
            "ba": ba_log,
        }

    def _correspondences(self, u, poses, kp2pt):
        kps, pids = [], []
        for n, iu, in_ in self.g.nbrs[u]:
            if n not in poses:
                continue
            pid = kp2pt[n][in_]
            ok = pid >= 0
            kps.append(iu[ok])
            pids.append(pid[ok])
        if not kps:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        key = np.unique(np.stack([np.concatenate(kps), np.concatenate(pids)], 1), axis=0)
        return key[:, 0], key[:, 1]

    def _count_inliers(self, R, t, xyz, obs):
        uv, z = project(R, t, xyz)
        err = np.linalg.norm(uv - obs, axis=1) * self.f
        return int(((err <= self.cfg.inlier_px) & (z > 0)).sum())

    def _start_poses(self, u, poses):
        """Starting poses for image u: its prior, and the prior carried over from the registered neighbours.

        The relative motion between close frames is far more accurate than the (drifting) global ARKit pose, so
        T_u = T_u^prior (T_n^prior)^-1 T_n^refined inherits the correction already found for a neighbour n.
        """
        starts = [self._rt(u)]
        prior_u = self.prior[u]
        neighbours = sorted((x for x in self.g.nbrs[u] if x[0] in poses), key=lambda x: -len(x[1]))
        for n, _, _ in neighbours[:3]:
            Rn, tn = poses[n]
            T = prior_u @ np.linalg.inv(self.prior[n]) @ pose_matrix(Rn, tn)
            starts.append((T[:3, :3].copy(), T[:3, 3].copy()))
        return starts

    def _register_image(self, u, corr, xyz_all, poses):
        """Stage 2: refine the pose of image u from its 2D-3D correspondences (LM on the reprojection error)."""
        cfg = self.cfg
        kp, pid = corr
        obs_np, pts_np = self.g.xn[u][kp], xyz_all[pid]
        obs, pts = torch.from_numpy(obs_np), torch.from_numpy(pts_np)
        starts = self._start_poses(u, poses)
        counts = [self._count_inliers(R, t, pts_np, obs_np) for R, t in starts]
        Rp, tp = starts[int(np.argmax(counts))]
        prior_inliers = max(counts)

        def solve(R, t):
            info = {}
            for th in self._thresholds():
                with torch.no_grad():
                    err = reprojection_px(R, t, pts, obs, self.f).norm(dim=1)
                    ok = (err <= th) & ((pts @ R.T + t)[:, 2] > 0)
                sel = ok.nonzero().squeeze(1)
                if len(sel) < 6:
                    return None
                R, t, info = robust_lm(
                    lambda R_, t_: reprojection_px(R_, t_, pts[sel], obs[sel], self.f),
                    R, t, huber=min(cfg.huber_px, th), max_iters=cfg.max_iters, unit_translation=False,
                )
            return R.numpy(), t.numpy()

        out = solve(torch.from_numpy(Rp), torch.from_numpy(tp))
        if out is None or self._count_inliers(*out, pts_np, obs_np) < cfg.min_inliers:
            out = self._ransac_pose(u, pts_np, obs_np)
            if out is not None:
                out = solve(torch.from_numpy(out[0]), torch.from_numpy(out[1]))
        if out is None:
            return None
        inliers = self._count_inliers(*out, pts_np, obs_np)
        if inliers < cfg.min_inliers or inliers < prior_inliers:
            return None
        return out[0], out[1], inliers

    def _ransac_pose(self, u, pts, obs):
        import pycolmap

        px = obs * self.f
        cam = pycolmap.Camera(model="PINHOLE", width=2, height=2, params=[self.f, self.f, 0.0, 0.0])
        options = pycolmap.AbsolutePoseEstimationOptions()
        options.ransac.max_error = self.cfg.inlier_px * 2
        try:
            res = pycolmap.estimate_and_refine_absolute_pose(px, pts, cam, options)
        except Exception:
            return None
        if not res:
            return None
        m = res["cam_from_world"].matrix()
        return m[:3, :3].copy(), m[:3, 3].copy()

    def _extend_map(self, a, b, ia, ib, poses, kp2pt, add_points, get_xyz):
        """Triangulate new points / extend tracks from the matches (ia, ib) of two registered images."""
        cfg, g = self.cfg, self.g
        pa, pb = kp2pt[a][ia], kp2pt[b][ib]
        Ra, ta = poses[a]
        Rb, tb = poses[b]
        xyz = get_xyz()

        new = (pa < 0) & (pb < 0)
        if new.any():
            x0, x1 = g.xn[a][ia[new]], g.xn[b][ib[new]]
            pts = triangulate_dlt(proj_matrix(Ra, ta), proj_matrix(Rb, tb), x0, x1)
            ok = np.isfinite(pts).all(1)
            pts = np.where(ok[:, None], pts, 0.0)
            ua, za = project(Ra, ta, pts)
            ub, zb = project(Rb, tb, pts)
            err = np.maximum(np.linalg.norm(ua - x0, axis=1), np.linalg.norm(ub - x1, axis=1)) * self.f
            angle = triangulation_angle_deg(-Ra.T @ ta, -Rb.T @ tb, pts)
            ok &= (za > 0) & (zb > 0) & (err <= cfg.new_point_max_reproj_px) & (angle >= cfg.new_point_min_angle_deg)
            sel = np.flatnonzero(new)[ok]
            ids = add_points(pts[ok])
            kp2pt[a][ia[sel]] = ids
            kp2pt[b][ib[sel]] = ids

        for src, dst, i_src, i_dst, (Rd, td) in ((a, b, ia, ib, poses[b]), (b, a, ib, ia, poses[a])):
            has, lacks = kp2pt[src][i_src], kp2pt[dst][i_dst]
            ext = np.flatnonzero((has >= 0) & (lacks < 0))
            if len(ext) == 0:
                continue
            xyz = get_xyz()
            uv, z = project(Rd, td, xyz[has[ext]])
            err = np.linalg.norm(uv - g.xn[dst][i_dst[ext]], axis=1) * self.f
            keep = (z > 0) & (err <= cfg.new_point_max_reproj_px)
            kp2pt[dst][i_dst[ext[keep]]] = has[ext[keep]]

    def _bundle_adjust(self, poses, kp2pt, xyz, *, anchors, free, global_ba):
        """Bundle adjustment (COLMAP / Ceres, intrinsics fixed) of the current incremental map.

        local  (`free` = recently registered images): only those poses and the points they observe are variable;
               every other image that sees these points stays fixed and anchors the solution (as in COLMAP's
               incremental mapper). The two initial images are always fixed here.
        global (`global_ba`): all poses and the `ba_max_points` longest tracks are variable; the gauge is fixed on
               the two initial images, which keeps the map in the frame and scale of the prior.
        `poses` / `xyz` are updated in place; observations above `ba_max_reproj_px` are removed from `kp2pt`.
        """
        import pycolmap

        cfg, g = self.cfg, self.g
        t_start = time.time()
        fx, fy, cx, cy = g.K
        scale = np.array([fx, fy])
        centre = np.array([cx, cy])
        n_pts = len(xyz)

        # ---- which points are optimised
        counts = np.zeros(n_pts, dtype=np.int64)
        for n in poses:
            p = kp2pt[n]
            counts += np.bincount(p[p >= 0], minlength=n_pts)
        mask = np.zeros(n_pts, dtype=bool)
        if global_ba:
            mask = counts >= 2
            if mask.sum() > cfg.ba_max_points:
                idx = np.flatnonzero(mask)
                order = np.random.default_rng(0).permutation(len(idx))
                idx = idx[order][np.argsort(-counts[idx][order], kind="stable")][: cfg.ba_max_points]
                mask = np.zeros(n_pts, dtype=bool)
                mask[idx] = True
            free_names = set(poses)
        else:
            free_names = set(free) - set(anchors)
            for n in free_names:
                p = kp2pt[n]
                mask[p[p >= 0]] = True
            mask &= counts >= 2

        # ---- observations of those points in every registered image
        obs = {}
        for n in poses:
            pid = kp2pt[n]
            ok = pid >= 0
            ok[ok] = mask[pid[ok]]
            kps = np.flatnonzero(ok)
            if len(kps):
                obs[n] = (kps, pid[kps])
        if not obs or not free_names:
            return None
        per_point = np.bincount(np.concatenate([p for _, p in obs.values()]), minlength=n_pts)
        obs = {n: (k[per_point[p] >= 2], p[per_point[p] >= 2]) for n, (k, p) in obs.items()}
        obs = {n: v for n, v in obs.items() if len(v[0])}
        if not any(n in obs for n in free_names):
            return None

        # ---- sub-reconstruction
        names = [n for n in (anchors[0], anchors[1]) if n in obs] + [n for n in obs if n not in anchors]
        rec = pycolmap.Reconstruction()
        width, height = int(round(2 * cx)), int(round(2 * cy))
        rec.add_camera_with_trivial_rig(
            pycolmap.Camera(model="PINHOLE", width=width, height=height, params=[fx, fy, cx, cy], camera_id=1)
        )
        image_id = {}
        for iid, n in enumerate(names, start=1):
            kps, _ = obs[n]
            R, t = poses[n]
            image = pycolmap.Image(name=n, keypoints=g.xn[n][kps] * scale + centre, camera_id=1, image_id=iid)
            rec.add_image_with_trivial_frame(image, pycolmap.Rigid3d(pose_matrix(R, t)[:3]))
            image_id[n] = iid
        all_pid = np.concatenate([obs[n][1] for n in names])
        all_iid = np.concatenate([np.full(len(obs[n][1]), image_id[n]) for n in names])
        all_idx = np.concatenate([np.arange(len(obs[n][1])) for n in names])
        order = np.argsort(all_pid, kind="stable")
        all_pid, all_iid, all_idx = all_pid[order], all_iid[order], all_idx[order]
        uniq, start = np.unique(all_pid, return_index=True)
        stop = np.append(start[1:], len(all_pid))
        new_ids = []
        for p, s, e in zip(uniq, start, stop):
            track = pycolmap.Track()
            for k in range(s, e):
                track.add_element(int(all_iid[k]), int(all_idx[k]))
            new_ids.append(rec.add_point3D(xyz[p], track))

        def residual_stats():
            errs = []
            for n in names:
                kps, pid = obs[n]
                R, t = poses[n]
                uv, z = project(R, t, xyz[pid])
                errs.append(np.where(z > 0, np.linalg.norm(uv - g.xn[n][kps], axis=1) * self.f, np.inf))
            return errs

        errs0 = np.concatenate(residual_stats())
        before = {n: (poses[n][0].copy(), -poses[n][0].T @ poses[n][1]) for n in names}

        # ---- solve
        config = pycolmap.BundleAdjustmentConfig()
        for n in names:
            config.add_image(image_id[n])
        config.set_constant_cam_intrinsics(1)
        for n in names:
            if n not in free_names:
                config.set_constant_rig_from_world_pose(rec.images[image_id[n]].frame_id)
        if global_ba:
            config.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
        options = pycolmap.BundleAdjustmentOptions()
        options.refine_focal_length = False
        options.refine_principal_point = False
        options.refine_extra_params = False
        options.refine_sensor_from_rig = False
        options.refine_rig_from_world = True
        options.refine_points3D = True
        options.print_summary = False
        options.ceres.loss_function_type = pycolmap.LossFunctionType.HUBER
        options.ceres.loss_function_scale = cfg.ba_huber_px
        options.ceres.auto_select_solver_type = False
        solver = options.ceres.solver_options
        solver.linear_solver_type = type(solver.linear_solver_type).SPARSE_SCHUR
        solver.max_num_iterations = cfg.ba_max_iters_global if global_ba else cfg.ba_max_iters_local
        summary = pycolmap.create_default_bundle_adjuster(options, config, rec).solve()
        if not summary.is_solution_usable():
            logger.warning(f"BA failed: {summary.brief_report()}")
            return None

        # ---- copy back, then drop observations that are still bad
        for n in free_names:
            if n in image_id:
                m = rec.images[image_id[n]].cam_from_world().matrix()
                poses[n] = (m[:3, :3].copy(), m[:3, 3].copy())
        for p, pid_new in zip(uniq, new_ids):
            xyz[p] = rec.points3D[pid_new].xyz
        rot_move, pos_move = [], []
        for n in names:
            if n in free_names:
                R, t = poses[n]
                rot_move.append(rotation_angle_deg(R @ before[n][0].T))
                pos_move.append(np.linalg.norm(-R.T @ t - before[n][1]))
        errs1 = residual_stats()
        for n, e in zip(names, errs1):
            bad = e > cfg.ba_max_reproj_px
            if bad.any():
                kp2pt[n][obs[n][0][bad]] = -1
        e1 = np.concatenate(errs1)
        return {
            "points": int(len(uniq)),
            "observations": int(len(e1)),
            "reproj_before": float(np.mean(np.minimum(errs0, 10.0))),
            "reproj_after": float(np.mean(np.minimum(e1, 10.0))),
            "rot_move_deg": float(np.mean(rot_move)) if rot_move else 0.0,
            "pos_move_cm": float(np.mean(pos_move) * 100) if pos_move else 0.0,
            "removed_obs": int((e1 > cfg.ba_max_reproj_px).sum()),
            "seconds": time.time() - t_start,
        }


def proj_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.concatenate([R, t[:, None]], axis=1)


def similarity_fit(src: np.ndarray, dst: np.ndarray, iters: int = 3):
    """Robust Sim(3) with dst ~= s R src + t (Umeyama, re-fitted on the points within 2x the median residual)."""
    keep = np.ones(len(src), bool)
    for _ in range(iters):
        a, b = src[keep], dst[keep]
        mu_a, mu_b = a.mean(0), b.mean(0)
        xa, xb = a - mu_a, b - mu_b
        u, sig, vt = np.linalg.svd(xb.T @ xa / len(a))
        d = np.diag([1.0, 1.0, np.sign(np.linalg.det(u) * np.linalg.det(vt))])
        R = u @ d @ vt
        s = np.trace(np.diag(sig) @ d) / xa.var(0).sum()
        t = mu_b - s * R @ mu_a
        res = np.linalg.norm(s * src @ R.T + t - dst, axis=1)
        keep = res <= max(2.0 * np.median(res), 1e-6)
        if keep.sum() < 4:
            break
    return s, R, t


def align_to_prior(poses: dict[str, np.ndarray], prior: dict[str, np.ndarray], names: list[str]):
    """Express the refined poses in the frame and metric scale of the prior (gauge fixing).

    The refinement is anchored on the first camera of a possibly short-baseline initial pair, so its absolute
    scale is only as good as that baseline. The relative geometry (what the refinement improves) is kept, the
    overall similarity is taken from all cameras of the (metric, gravity-aligned) prior.
    """
    centre = lambda m: -m[:3, :3].T @ m[:3, 3]
    src = np.array([centre(poses[n]) for n in names])
    dst = np.array([centre(prior[n]) for n in names])
    s, Rg, tg = similarity_fit(src, dst)
    out = {}
    for n in names:
        R = poses[n][:3, :3] @ Rg.T
        out[n] = pose_matrix(R, -R @ (s * Rg @ centre(poses[n]) + tg))
    return out, float(s)


def pose_change(poses: dict[str, np.ndarray], ref: dict[str, np.ndarray]) -> dict:
    shifts, angles = [], []
    for n, p in poses.items():
        c0, c1 = -ref[n][:3, :3].T @ ref[n][:3, 3], -p[:3, :3].T @ p[:3, 3]
        shifts.append(np.linalg.norm(c1 - c0))
        angles.append(rotation_angle_deg(p[:3, :3] @ ref[n][:3, :3].T))
    shifts, angles = np.array(shifts), np.array(angles)
    return {
        "pos_mean_cm": float(shifts.mean() * 100),
        "pos_max_cm": float(shifts.max() * 100),
        "rot_mean_deg": float(angles.mean()),
        "rot_max_deg": float(angles.max()),
    }


def refine_prior_poses(
    feat_path: Path,
    match_path: Path,
    pairs_path: Path,
    names: list[str],
    poses: dict[str, np.ndarray],
    intrinsics: list[float],
    cfg: RefineConfig | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Return (refined cam_from_world 4x4 per image, statistics) for prior poses `poses` (cam_from_world)."""
    cfg = cfg or RefineConfig()
    graph = MatchGraph(feat_path, match_path, pairs_path, names, intrinsics)
    refiner = PoseRefiner(graph, names, poses, cfg)
    refined, stats = refiner.run()
    stats["config"] = asdict(cfg)
    stats["epipolar_prior"] = epipolar_stats(graph, poses)
    stats["epipolar_refined"] = epipolar_stats(graph, refined)
    stats["change_vs_prior"] = pose_change(refined, poses)
    for tag in ("prior", "refined"):
        logger.info(f"epipolar consistency of the {tag} poses: {stats['epipolar_' + tag]}")
    logger.info(f"refined vs prior poses: {stats['change_vs_prior']}")
    return refined, stats
