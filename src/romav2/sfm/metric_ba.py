"""Joint sparse bundle adjustment with fixed pinhole calibration and metric pose priors."""

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation


# Dense, pairwise matches in the same image share sources of error. Cap their
# total weight so denser sampling does not silently weaken a pose measurement.
MAX_VISUAL_WEIGHT_PER_IMAGE = 16


def _skew(v):
    x, y, z = v.T
    o = np.zeros_like(x)
    return np.stack((o, -z, y, z, o, -x, -y, x, o), axis=1).reshape(-1, 3, 3)


def adjust(model, images, priors, rotation_sigma, pixel_sigma, iterations):
    import pycolmap

    ids = list(model.points3D)
    indices = {key: j for j, key in enumerate(ids)}
    xyz = np.array([model.points3D[key].xyz for key in ids])
    observations = [
        (i, indices[p.point3D_id], p.xy)
        for i, im in enumerate(images)
        for p in im.points2D
        if p.has_point3D()
    ]
    if not observations:
        return False
    view, point, pixels = map(np.asarray, zip(*observations))
    k = np.array([model.cameras[im.camera_id].params for im in images])[view]
    focal, principal = k[:, :2], k[:, 2:]
    rotations = np.array([im.cam_from_world().rotation.matrix() for im in images])
    centers = np.array([im.projection_center() for im in images])
    prior = np.array([priors.poses[im.name] for im in images])
    n = len(images)
    counts = np.bincount(view, minlength=n)
    visual_weight = (
        MAX_VISUAL_WEIGHT_PER_IMAGE / np.maximum(counts, MAX_VISUAL_WEIGHT_PER_IMAGE)
    )[view]
    rows = np.arange(2 * len(view)).reshape(-1, 2, 1)
    columns = np.column_stack(
        (6 * view[:, None] + np.arange(6), 6 * n + 3 * point[:, None] + np.arange(3))
    )
    row = np.broadcast_to(rows, (len(view), 2, 9)).ravel()
    col = np.broadcast_to(columns[:, None], (len(view), 2, 9)).ravel()
    pose_rows = rows.size + np.arange(6 * n).reshape(n, 6, 1)
    pose_cols = np.arange(6 * n).reshape(n, 1, 6)
    row = np.r_[row, np.broadcast_to(pose_rows, (n, 6, 6)).ravel()]
    col = np.r_[col, np.broadcast_to(pose_cols, (n, 6, 6)).ravel()]

    def evaluate(rot, center, points, jacobian=False):
        camera = (rot[view] @ (points[point] - center[view])[:, :, None])[:, :, 0]
        if np.any(camera[:, 2] <= 1e-8):
            return None
        uv = camera[:, :2] / camera[:, 2:]
        orientation = Rotation.from_matrix(rot @ prior[:, :3, :3]).as_rotvec()
        res = np.r_[
            ((uv * focal + principal - pixels) / pixel_sigma).ravel(),
            np.column_stack(
                (
                    orientation / rotation_sigma,
                    (center - prior[:, :3, 3]) / priors.position_sigma,
                )
            ).ravel(),
        ]
        if not jacobian:
            return res
        projection = np.empty((len(view), 2, 3))
        projection[:, :, :2] = np.eye(2)
        projection[:, :, 2] = -uv
        projection *= (focal / (pixel_sigma * camera[:, 2:]))[:, :, None]
        world = projection @ rot[view]
        visual = np.concatenate((-projection @ _skew(camera), -world, world), axis=2)
        # Differential of log(exp(delta) R) for the left rotation update.
        angle = np.linalg.norm(orientation, axis=1).clip(1e-5)
        hat = _skew(orientation)
        coeff = (1 - 0.5 * angle / np.tan(0.5 * angle)) / angle**2
        pose = np.zeros((n, 6, 6))
        pose[:, :3, :3] = (
            np.eye(3) - 0.5 * hat + coeff[:, None, None] * (hat @ hat)
        ) / rotation_sigma
        pose[:, 3:, 3:] = np.eye(3) / priors.position_sigma
        jac = coo_matrix(
            (np.r_[visual.ravel(), pose.ravel()], (row, col)),
            shape=(len(res), 6 * n + xyz.size),
        ).tocsr()
        return res, jac

    def loss(residual):
        pixels = residual[: rows.size].reshape(-1, 2)
        pixel_scale = np.sqrt(1 + np.sum(pixels**2, axis=1))
        pose = residual[rows.size :]
        cost = (visual_weight @ (pixel_scale - 1)) + 0.5 * (pose @ pose)
        return cost, pixel_scale

    damping = 1e-3
    accepted = False
    for _ in range(10 * iterations):
        result = evaluate(rotations, centers, xyz, True)
        if result is None:
            return False
        residual, jac = result
        cost, pixel_scale = loss(residual)
        weight = np.r_[np.repeat(visual_weight / pixel_scale, 2), np.ones(6 * n)]
        normal = jac.T @ jac.multiply(weight[:, None])
        system = (normal + diags(damping * normal.diagonal().clip(1e-8))).tocsr()
        rhs = -jac.T @ (weight * residual)
        # Each landmark contributes a single independent 3x3 diagonal block.
        cut = 6 * n
        cross = system[:cut, cut:]
        inverse = system[cut:, cut:].tobsr(blocksize=(3, 3))
        inverse.data[:] = np.linalg.inv(inverse.data)
        reduced = cross @ inverse
        poses_step = spsolve(
            (system[:cut, :cut] - reduced @ cross.T).tocsc(),
            rhs[:cut] - reduced @ rhs[cut:],
        )
        step = np.r_[poses_step, inverse @ (rhs[cut:] - cross.T @ poses_step)]
        if not np.isfinite(step).all():
            break
        dr, dc = np.split(step[: 6 * n].reshape(n, 6), 2, axis=1)
        trial = (
            Rotation.from_rotvec(dr).as_matrix() @ rotations,
            centers + dc,
            xyz + step[6 * n :].reshape(-1, 3),
        )
        candidate = evaluate(*trial)
        if candidate is not None:
            gain = cost - loss(candidate)[0]
            if gain > 0:
                accepted = True
                rotations, centers, xyz = trial
                damping = max(1e-8, damping / 3)
                if gain < 1e-7 * max(1, cost):
                    break
                continue
        damping *= 10
        if damping > 1e8:
            break
    if not accepted:
        return False
    for im, r, c in zip(images, rotations, centers):
        model.frames[im.frame_id].set_cam_from_world(
            im.camera_id, pycolmap.Rigid3d(pycolmap.Rotation3d(r), -r @ c)
        )
    for key, value in zip(ids, xyz):
        model.points3D[key].xyz = value
    return True
