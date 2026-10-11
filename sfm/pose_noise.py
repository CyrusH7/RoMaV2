"""Synthetic ARKit-like errors for ground-truth camera poses (used to build coarse pose priors from GT).

Real VIO poses (ARKit) are not i.i.d. noisy: they drift. The perturbation therefore has three parts, applied to the
camera-to-world pose of every frame, in temporal (sorted-name) order:

    iid     independent rotation (deg, std of the axis-angle vector per axis) and camera-centre (cm) noise
    drift   random walk, one step per kept frame: rotation (deg) and camera-centre (cm) increments that accumulate
    scale   global scale error of the trajectory (fraction, e.g. 0.03 = 3%) about the first camera centre

PRESETS are rough "low / mid / high" ARKit-quality levels for a ~100-frame sequence of a room-scale scene.
Poses are 4x4 cam_from_world matrices (COLMAP convention) everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class NoiseConfig:
    rot_deg: float = 0.0
    trans_cm: float = 0.0
    drift_rot_deg: float = 0.0
    drift_trans_cm: float = 0.0
    scale_error: float = 0.0
    seed: int = 0

    def is_zero(self) -> bool:
        return not any((self.rot_deg, self.trans_cm, self.drift_rot_deg, self.drift_trans_cm, self.scale_error))


PRESETS = {
    "none": NoiseConfig(),
    "low": NoiseConfig(rot_deg=0.2, trans_cm=0.5, drift_rot_deg=0.05, drift_trans_cm=0.2, scale_error=0.0),
    "mid": NoiseConfig(rot_deg=0.5, trans_cm=2.0, drift_rot_deg=0.1, drift_trans_cm=0.5, scale_error=0.02),
    "high": NoiseConfig(rot_deg=1.0, trans_cm=5.0, drift_rot_deg=0.2, drift_trans_cm=1.0, scale_error=0.05),
}


def so3_exp(w: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    k = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]) / theta
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * k @ k


def perturb_poses(poses: dict[str, np.ndarray], cfg: NoiseConfig) -> dict[str, np.ndarray]:
    """Perturb cam_from_world poses (dict keyed by image stem / name, temporal order = sorted keys)."""
    rng = np.random.default_rng(cfg.seed)
    names = sorted(poses)
    centers = np.array([-poses[n][:3, :3].T @ poses[n][:3, 3] for n in names])
    rot_wc = [poses[n][:3, :3].T for n in names]  # world_from_cam
    n = len(names)

    w_iid = np.radians(rng.normal(0.0, cfg.rot_deg, (n, 3)))
    c_iid = rng.normal(0.0, cfg.trans_cm / 100.0, (n, 3))
    w_drift = np.cumsum(np.radians(rng.normal(0.0, cfg.drift_rot_deg, (n, 3))), axis=0)
    c_drift = np.cumsum(rng.normal(0.0, cfg.drift_trans_cm / 100.0, (n, 3)), axis=0)

    centers = centers[0] + (1.0 + cfg.scale_error) * (centers - centers[0])
    out = {}
    for i, name in enumerate(names):
        r_wc = so3_exp(w_iid[i] + w_drift[i]) @ rot_wc[i]
        c = centers[i] + c_iid[i] + c_drift[i]
        m = np.eye(4)
        m[:3, :3] = r_wc.T
        m[:3, 3] = -r_wc.T @ c
        out[name] = m
    return out


def load_poses_yaml(path: Path) -> dict[str, np.ndarray]:
    """camera_poses.yaml -> {image stem: 4x4 cam_from_world}."""
    data = yaml.safe_load(Path(path).read_text())["camera_poses"]
    return {Path(k).stem: np.array(v["transform_matrix"], dtype=np.float64) for k, v in data.items()}


def save_poses_yaml(poses: dict[str, np.ndarray], path: Path, images_dir_name: str = "images") -> None:
    payload = {
        "camera_poses": {f"{images_dir_name}/{stem}": {"transform_matrix": poses[stem].tolist()} for stem in sorted(poses)}
    }
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False))
