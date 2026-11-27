"""Metric pose initialization from raw calibrated correspondences, before inlier selection."""

from dataclasses import replace

import numpy as np
import pycolmap
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, eye, kron, vstack
from scipy.spatial.transform import Rotation


def refine_pose_graph(path, pairs, ids, priors, options):
    names = sorted(ids)
    index = {name: i for i, name in enumerate(names)}
    poses = np.stack([priors.poses[name] for name in names])
    rng = np.random.default_rng(options.seed % 2**32)
    fit_count, held_count = 48, 16
    sample_count = fit_count + held_count
    edges, samples = [], []
    with pycolmap.Database.open(str(path)) as db:
        cameras = {im.name: db.read_camera(im.camera_id) for im in db.read_all_images()}
        pixels = {n: db.read_keypoints(ids[n])[:, :2] for n in names}
        for a, b in pairs:
            i, j = index[a], index[b]
            if (
                np.linalg.norm(poses[i, :3, 3] - poses[j, :3, 3])
                <= 2 * priors.position_sigma
            ):
                continue
            matches = db.read_matches(ids[a], ids[b])
            if len(matches) < max(sample_count, options.min_matches):
                continue
            chosen = matches[rng.choice(len(matches), sample_count, replace=False)]
            edges.append((i, j))
            samples.append(
                [
                    np.column_stack(
                        (
                            cameras[name].cam_from_img(pixels[name][idx]),
                            np.ones(sample_count),
                        )
                    )
                    for name, idx in zip((a, b), chosen.T)
                ]
            )
    report = dict(mode="graph", status="kept_input_poses", pairs=len(edges))
    if len(edges) < 3:
        return priors, {**report, "reason": "insufficient_pairs"}

    a, b = np.asarray(edges).T
    rays_a, rays_b = np.stack(samples, axis=1)
    focal = np.array([cameras[name].params[:2] for name in names])
    sigma_r = np.deg2rad(priors.rotation_sigma_deg)

    def unpack(step):
        rotation, offset = np.split(step.reshape(-1, 6), 2, axis=1)
        return (
            Rotation.from_rotvec(sigma_r * rotation).as_matrix() @ poses[:, :3, :3],
            poses[:, :3, 3] + priors.position_sigma * offset,
        )

    def epipolar(step, subset, check_depth=False):
        rotation, center = unpack(step)
        world_to_b = rotation[b].transpose(0, 2, 1)
        relative = world_to_b @ rotation[a]
        t = (world_to_b @ (center[a] - center[b])[:, :, None]).transpose(0, 2, 1)
        essential = np.cross(t, relative.transpose(0, 2, 1)).transpose(0, 2, 1)
        x, y = rays_a[:, subset], rays_b[:, subset]
        line_a, line_b = y @ essential, x @ essential.transpose(0, 2, 1)
        norm_a = np.sum((line_a[:, :, :2] / focal[a, None]) ** 2, axis=2)
        norm_b = np.sum((line_b[:, :, :2] / focal[b, None]) ** 2, axis=2)
        normal = np.sqrt(norm_a + norm_b)
        numerator = np.sum(y * line_b, axis=2)
        error = np.divide(
            numerator,
            normal,
            out=np.full_like(normal, 2 * options.geometric_max_error),
            where=normal > 1e-14,
        )
        if check_depth:
            rotated = x @ relative.transpose(0, 2, 1)
            cross = np.cross(rotated, y)
            positive = (np.sum(np.cross(y, t) * cross, axis=2) > 0) & (
                np.sum(np.cross(rotated, t) * cross, axis=2) > 0
            )
            symmetric_normal = np.sqrt(np.minimum(norm_a, norm_b))
            supported = (
                positive
                & (symmetric_normal > 1e-14)
                & (abs(numerator) <= options.geometric_max_error * symmetric_normal)
            )
            score = np.where(
                positive,
                np.minimum(abs(error), options.geometric_max_error),
                options.geometric_max_error,
            )
            return float(score.mean()), supported.mean(axis=1)
        return error

    def residual(step):
        error = epipolar(step, slice(fit_count))
        # Soft-L1 in pixels; cap each pair at eight effective independent matches.
        visual = error * np.sqrt((8 / fit_count) * 2 / (1 + np.hypot(1, error)))
        return np.r_[visual.ravel(), step]

    incidence = csr_matrix(
        (
            np.ones(2 * len(edges)),
            (np.repeat(np.arange(len(edges)), 2), np.ravel(edges)),
        ),
        shape=(len(edges), len(names)),
    )
    start = np.zeros(6 * len(names))
    held = slice(fit_count, None)
    before, support = epipolar(start, held, check_depth=True)
    report["prior_supported_pair_fraction"] = float(np.mean(support > 0.5))
    if report["prior_supported_pair_fraction"] > 0.5:
        return priors, {**report, "reason": "raw_matches_support_input_poses"}
    structure = vstack(
        (kron(incidence, np.ones((fit_count, 6))), eye(start.size)), format="csr"
    )
    fit = least_squares(
        residual, start, jac_sparsity=structure, bounds=(-3, 3), max_nfev=30, ftol=1e-5
    )
    after, _ = epipolar(fit.x, held, check_depth=True)
    report.update(
        held_out_score_before_px=before,
        held_out_score_after_px=after,
        optimizer_status=fit.status,
    )
    initial_residual = residual(start)
    if fit.status < 0 or not (
        after < before and 2 * fit.cost < initial_residual @ initial_residual
    ):
        return priors, {**report, "reason": "fit_or_validation_failed"}
    updated = poses.copy()
    updated[:, :3, :3], updated[:, :3, 3] = unpack(fit.x)
    return replace(priors, poses=dict(zip(names, updated))), {
        **report,
        "status": "refined",
    }
