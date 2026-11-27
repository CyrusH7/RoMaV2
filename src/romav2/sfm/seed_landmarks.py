"""Metric seed-pair refinement using tracks triangulated outside the pair."""

import numpy as np
import pycolmap
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def sampson_pixels(rotation, direction, rays_a, rays_b, focal_a, focal_b):
    essential = np.cross(direction, rotation.T).T
    line_a = rays_b @ essential
    line_b = rays_a @ essential.T
    normal = np.linalg.norm(
        np.column_stack((line_a[:, :2] / focal_a, line_b[:, :2] / focal_b)), axis=1
    )
    numerator = np.sum(rays_b * line_b, axis=1)
    return numerator / np.maximum(normal, 1e-15)


def _reference_points(tracks, count):
    # Intersect reference rays in the prior world frame. A seed observation
    # identifies a track but contributes no ray to its 3D estimate.
    normals = np.zeros((count, 3, 3))
    rhs = np.zeros((count, 3))
    views = np.zeros(count, dtype=int)
    for pose, indices, xy, _ in tracks:
        rays = np.column_stack((xy, np.ones(len(xy)))) @ pose.rotation.matrix()
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        projection = np.eye(3) - rays[:, :, None] * rays[:, None, :]
        normals[indices] += projection
        rhs[indices] += projection @ pose.inverse().translation
        views[indices] += 1
    spectrum = np.linalg.eigvalsh(normals)
    good = (views >= 3) & (spectrum[:, 0] > views * np.sin(np.deg2rad(1.5)) ** 2)
    xyz = np.zeros_like(rhs)
    xyz[good] = np.linalg.solve(normals[good], rhs[good, :, None])[:, :, 0]
    for pose, indices, _, _ in tracks:
        good[indices] &= (pose * xyz[indices])[:, 2] > 0
    covariance = np.zeros_like(normals)
    covariance[good] = np.linalg.inv(normals[good])
    return xyz, good, covariance


def refine_landmark_pair(
    poses, observations, tracks, held, sigma_r, sigma_c, threshold, min_support
):
    """Estimate metric relative motion with bidirectional, uncertainty-weighted pixel transfer."""
    a, b = poses
    ra, rb, fa, fb = observations
    xyz, good, covariance = _reference_points(tracks, len(ra))
    za, zb = (a * xyz)[:, 2], (b * xyz)[:, 2]
    good &= (za > 0) & (zb > 0)
    test = np.zeros(len(ra), dtype=bool)
    test[held] = True
    train, test = good & ~test, good & test
    details = dict(
        reference_tracks_fit=int(train.sum()), reference_tracks_held=int(test.sum())
    )
    if train.sum() < min_support or test.sum() < 8:
        return None, {**details, "reason": "insufficient_reference_tracks"}
    prior = b * a.inverse()
    r0 = prior.rotation.matrix()
    t0 = prior.translation
    zero = np.zeros(6)

    def motion(step):
        r = Rotation.from_rotvec(sigma_r * step[:3]).as_matrix() @ r0
        return r, t0 + sigma_c * step[3:]

    def depth_uncertainty(pose, depth):
        axis = pose.rotation.matrix()[2]
        return np.sqrt(
            (sigma_c**2 + (sigma_r * depth) ** 2) * (axis @ covariance @ axis)
            + sigma_c**2
        )

    ua, ub = depth_uncertainty(a, za), depth_uncertainty(b, zb)

    def transfer(step, mask, whiten=False):
        r, t = motion(step)

        def project(ray, depth, sigma, rot, trans, target, focal):
            direction = ray[mask] @ rot.T
            point = depth[mask, None] * direction + trans
            z = np.maximum(point[:, 2:], 1e-6)
            xy = point[:, :2] / z
            err = (xy - target[mask, :2]) * focal
            if whiten:
                # A depth perturbation produces rank-one image covariance.
                gradient = (direction[:, :2] - xy * direction[:, 2:]) * focal / z
                v = gradient * sigma[mask, None]
                scale = np.sqrt(1 + np.sum(v * v, axis=1, keepdims=True))
                err -= (
                    v * np.sum(err * v, axis=1, keepdims=True) / (scale * (scale + 1))
                )
            return np.where(point[:, 2:] > 0, err, 1e4)

        return np.column_stack(
            (
                project(ra, za, ua, r, t, rb, fb),
                project(rb, zb, ub, r.T, -r.T @ t, ra, fa),
            )
        )

    def residual(step):
        errors = transfer(step, train, whiten=True)
        robust = errors * np.sqrt(2 / (np.hypot(1, errors / threshold) + 1))
        return np.r_[robust.ravel() / np.sqrt(train.sum()), step]

    def errors(step):
        err = transfer(step, test).reshape(-1, 2, 2)
        return np.linalg.norm(err, axis=2).mean(axis=1)

    before = errors(zero)
    fit = least_squares(residual, zero, max_nfev=50)
    if not fit.success or not np.isfinite(fit.x).all():
        return None, {**details, "reason": "solver_failed"}
    after = errors(fit.x)
    gain = before - after
    details.update(
        transfer_before_px=float(np.median(before)),
        transfer_after_px=float(np.median(after)),
    )
    if (
        np.median(gain) <= 1
        or details["transfer_before_px"] <= details["transfer_after_px"]
    ):
        return None, {**details, "reason": "heldout_transfer_rejected"}
    r, t = motion(fit.x)
    if np.linalg.norm(fit.x[:3]) > 2 or np.linalg.norm(fit.x[3:]) > 2 or t @ t0 <= 0:
        return None, {**details, "reason": "prior_step_rejected"}

    # Pick the world frame closest to both original orientations and centers.
    qa, qb = (Rotation.from_matrix(p.inverse().rotation.matrix()) for p in poses)
    rel = Rotation.from_matrix(r)
    correction = (qa.inv() * qb * rel).as_rotvec()
    qa = qa * Rotation.from_rotvec(0.5 * correction)
    qb = qa * rel.inv()
    midpoint = (a.inverse().translation + b.inverse().translation) / 2
    half = qb.apply(t) / 2
    matrices = []
    for q, center in ((qa, midpoint + half), (qb, midpoint - half)):
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = q.as_matrix(), center
        matrices.append(matrix)
    edge_errors = []
    for original, matrix, rays, focal in zip(poses, matrices, (ra, rb), (fa, fb)):
        refined = pycolmap.Rigid3d(matrix[:3]).inverse()
        per_view = []
        for ref, indices, xy, ref_focal in tracks:
            keep = test[indices]
            if keep.sum() < 3:
                continue
            observed = rays[np.asarray(indices)[keep]]
            reference = np.column_stack((xy[keep], np.ones(keep.sum())))
            errors = []
            for camera in (original, refined):
                relative = ref * camera.inverse()
                if np.linalg.norm(relative.translation) < 1e-9:
                    break
                pixel_errors = sampson_pixels(
                    relative.rotation.matrix(),
                    relative.translation,
                    observed,
                    reference,
                    focal,
                    ref_focal,
                )
                errors.append(np.median(np.abs(pixel_errors)))
            if len(errors) == 2:
                per_view.append(errors)
        if not per_view:
            return None, {**details, "reason": "insufficient_reference_edges"}
        edge_errors.append(np.median(per_view, axis=0))
    before, after = np.asarray(edge_errors).T
    details.update(
        reference_before_px=before.tolist(), reference_after_px=after.tolist()
    )
    if (
        before.mean() <= threshold
        or after.mean() >= before.mean()
        or np.any(after > before + 1)
    ):
        return None, {**details, "reason": "reference_edges_rejected"}
    return matrices, details
