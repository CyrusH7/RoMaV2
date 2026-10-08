"""Compare COLMAP sparse models: size, track quality, intrinsics, and agreement with the ARKit poses.

Usage:
    python sfm/compare_models.py --poses_yaml local/example/camera_poses.yaml \
        run_a=local/example/sparse/0 run_b=local/example_baseline/sparse/0

The ARKit poses are a drifting VIO estimate, not ground truth: "ATE"/"rot err" measure agreement with
them after a Sim(3) fit of the camera centers, so lower is closer to ARKit, not necessarily better.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pycolmap
import yaml


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Sim(3) with dst ~= s * R @ src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(xd.T @ xs / len(src))
    d = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[2, 2] = -1
    r = u @ d @ vt
    s = np.trace(np.diag(sig) @ d) / xs.var(0).sum()
    return s, r, mu_d - s * r @ mu_s


def rot_angle_deg(r: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def load_prior(path: Path) -> dict[str, np.ndarray]:
    data = yaml.safe_load(Path(path).read_text())["camera_poses"]
    return {Path(k).stem: np.array(v["transform_matrix"]) for k, v in data.items()}


def stats(rec: pycolmap.Reconstruction, prior: dict[str, np.ndarray] | None) -> dict:
    pts = list(rec.points3D.values())
    track = np.array([p.track.length() for p in pts]) if pts else np.zeros(0)
    err = np.array([p.error for p in pts]) if pts else np.zeros(0)
    focals = np.array([np.mean(c.focal_length_x if hasattr(c, "focal_length_x") else c.params[0:1])
                       for c in rec.cameras.values()])
    row = {
        "images": rec.num_reg_images(),
        "cameras": len(rec.cameras),
        "points": len(pts),
        "track>=3": int((track >= 3).sum()),
        "track>=5": int((track >= 5).sum()),
        "mean_track": float(track.mean()) if len(track) else 0.0,
        "reproj_px": float(err.mean()) if len(err) else 0.0,
        "focal_mean": float(focals.mean()),
        "focal_std": float(focals.std()),
    }
    if prior:
        cams, centers_prior, rots = [], [], []
        for image in rec.images.values():
            stem = Path(image.name).stem
            if stem not in prior or not image.has_pose:
                continue
            m = image.cam_from_world().matrix()
            cams.append((m[:3, :3], m[:3, 3]))
            centers_prior.append(-prior[stem][:3, :3].T @ prior[stem][:3, 3])
            rots.append(prior[stem][:3, :3])
        if len(cams) >= 3:
            centers_sfm = np.array([-r.T @ t for r, t in cams])
            centers_prior = np.array(centers_prior)
            s, r_a, t_a = umeyama(centers_sfm, centers_prior)
            aligned = s * (r_a @ centers_sfm.T).T + t_a
            ate = np.linalg.norm(aligned - centers_prior, axis=1)
            rot_err = [rot_angle_deg((rc @ r_a.T) @ rp.T) for (rc, _), rp in zip(cams, rots)]
            row.update(ate_cm=float(np.sqrt((ate**2).mean()) * 100), rot_err_deg=float(np.mean(rot_err)),
                       sfm_scale_vs_arkit=float(s))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", help="name=path/to/sparse/0")
    parser.add_argument("--poses_yaml", type=Path, default=None)
    args = parser.parse_args()

    prior = load_prior(args.poses_yaml) if args.poses_yaml else None
    rows = {}
    for item in args.models:
        name, _, path = item.partition("=")
        if not path or not Path(path).exists():
            print(f"skip {item}: model not found")
            continue
        rows[name] = stats(pycolmap.Reconstruction(path), prior)
    if not rows:
        raise SystemExit("no model to compare")

    keys = list(next(iter(rows.values())).keys())
    for r in rows.values():
        for k in r:
            if k not in keys:
                keys.append(k)
    width = max(len(n) for n in rows) + 2
    print("".ljust(18) + "".join(n.rjust(max(width, 12)) for n in rows))
    for k in keys:
        cells = []
        for r in rows.values():
            v = r.get(k)
            cells.append(("-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)).rjust(max(width, 12)))
        print(k.ljust(18) + "".join(cells))


if __name__ == "__main__":
    main()
