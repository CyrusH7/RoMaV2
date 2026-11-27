"""Register omitted prior-initialized views against the refined metric map."""

from collections import Counter, defaultdict

import numpy as np
import pycolmap
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def _correspondences(verified_pairs):
    point_ids = defaultdict(set)
    keypoint_ids = defaultdict(set)
    sources = defaultdict(set)
    for ref, matches in verified_pairs:
        for ref_idx, idx in matches:
            obs = ref.points2D[ref_idx]
            if obs.has_point3D():
                pid = obs.point3D_id
                point_ids[idx].add(pid)
                keypoint_ids[pid].add(idx)
                sources[idx, pid].add(ref.name)
    edges = [
        (i, p)
        for i, p in sources
        if len(point_ids[i]) == 1 and len(keypoint_ids[p]) == 1
    ]
    return edges, sources


def _reference_geometry(db, query, camera, keypoints, references, prior, options):
    geometry = pycolmap.TwoViewGeometryOptions()
    geometry.ransac.random_seed = options.seed
    geometry.ransac.max_error = options.geometric_max_error
    geometry.min_num_inliers = options.min_matches
    views, verified_pairs = [], []
    for ref in references:
        raw = db.read_matches(ref.image_id, query.image_id)
        if len(raw) < options.min_matches:
            continue
        if np.linalg.norm(ref.projection_center() - prior.inverse().translation) < 1e-8:
            continue
        ref_xy = db.read_keypoints(ref.image_id)[:, :2]
        verified = pycolmap.estimate_calibrated_two_view_geometry(
            ref.camera, ref_xy, camera, keypoints, raw, geometry
        ).inlier_matches
        if len(verified) < options.min_matches:
            continue
        verified_pairs.append((ref, verified))
        x = ref.camera.cam_from_img(ref_xy[verified[:, 0]])
        y = camera.cam_from_img(keypoints[verified[:, 1]])
        ones = np.ones(len(verified))
        views.append(
            (
                ref.cam_from_world().inverse(),
                np.column_stack((x, ones)),
                np.column_stack((y, ones)),
                ref.camera.params[:2],
            )
        )
    return views, verified_pairs


def _refine_pose(views, camera, prior, priors, xy, xyz, max_error, *, joint_prior_ba):
    def loss(squared):
        # Sampson and map residuals reuse matches against the fixed map.
        visual = squared[:-6]
        weight = min(1.0, 4 / len(visual))
        scale = np.sqrt(1 + visual)
        return np.array(
            [
                np.r_[2 * weight * (scale - 1), squared[-6:]],
                np.r_[weight / scale, np.ones(6)],
                np.r_[-0.5 * weight / scale**3, np.zeros(6)],
            ]
        )

    robust_loss = loss if joint_prior_ba else "soft_l1"

    def updated_pose(step):
        rotation = (
            pycolmap.Rotation3d(Rotation.from_rotvec(step[:3]).as_matrix())
            * prior.rotation
        )
        center = prior.inverse().translation + step[3:]
        return pycolmap.Rigid3d(rotation, -(rotation * center))

    def sampson(pose):
        residuals = []
        for world_from_ref, x, y, focal in views:
            relative = pose * world_from_ref
            if np.linalg.norm(relative.translation) < 1e-9:
                rotated = x @ relative.rotation.matrix().T
                displacement = (
                    rotated[:, :2] / np.maximum(rotated[:, 2:], 1e-8) - y[:, :2]
                ) * camera.params[:2]
                residuals.append(np.linalg.norm(displacement, axis=1))
                continue
            essential = pycolmap.essential_matrix_from_pose(relative)
            line_ref, line_query = y @ essential, x @ essential.T
            squared = np.sum(
                (line_ref[:, :2] / focal) ** 2
                + (line_query[:, :2] / camera.params[:2]) ** 2,
                axis=1,
            )
            residuals.append(
                np.sum(y * line_query, axis=1) / np.sqrt(np.maximum(squared, 1e-30))
            )
        return np.concatenate(residuals)

    def residual(step, use_map):
        pose = updated_pose(step)
        reprojection = []
        if len(use_map):
            local = pose * xyz[use_map]
            projected = local[:, :2] / np.maximum(local[:, 2:], 1e-8)
            reprojection = (
                projected * camera.params[:2] + camera.params[2:] - xy[use_map]
            ).ravel()
        return np.r_[
            sampson(pose),
            reprojection,
            step[:3] / np.deg2rad(priors.rotation_sigma_deg),
            step[3:] / priors.position_sigma,
        ]

    fit = least_squares(
        residual, np.zeros(6), args=([],), loss=robust_loss, max_nfev=60
    )
    if not fit.success or not np.isfinite(fit.x).all():
        return None
    pose = updated_pose(fit.x)
    local = pose * xyz
    errors = np.linalg.norm(camera.img_from_cam(local) - xy, axis=1)
    support = np.flatnonzero((local[:, 2] > 0) & (errors <= max_error))
    if len(support) >= 6:
        fit = least_squares(
            residual, fit.x, args=(support,), loss=robust_loss, max_nfev=60
        )
        if fit.success and np.isfinite(fit.x).all():
            pose = updated_pose(fit.x)
    return pose, float(np.median(np.abs(sampson(pose))))


def _localize(db, model, query, references, priors, options):
    prior = priors.cam_from_world(query.name)
    keypoints = db.read_keypoints(query.image_id)[:, :2]
    camera = db.read_camera(query.camera_id)
    views, pairs = _reference_geometry(
        db, query, camera, keypoints, references, prior, options
    )
    edges, sources = _correspondences(pairs)
    counts = dict(candidates=len(edges), status="insufficient_reference_geometry")
    if len(views) < 2:
        return None, counts
    counts["status"] = "insufficient_correspondences"
    mapper = pycolmap.IncrementalMapperOptions()
    if len(edges) < mapper.abs_pose_min_num_inliers:
        return None, counts
    xy = keypoints[[i for i, _ in edges]]
    xyz = np.asarray([model.points3D[p].xyz for _, p in edges])
    est = pycolmap.AbsolutePoseEstimationOptions()
    est.ransac.random_seed = options.seed
    est.ransac.max_error = min(
        options.geometric_max_error, mapper.filter_max_reproj_error
    )
    result = pycolmap.estimate_and_refine_absolute_pose(xy, xyz, camera, est)
    if result is None:
        counts["status"] = "pnp_failed"
        return None, counts
    counts["pnp_inliers"] = int(result["num_inliers"])
    if result["num_inliers"] < mapper.abs_pose_min_num_inliers:
        counts["status"] = "insufficient_inliers"
        return None, counts
    pnp_inliers = result["inlier_mask"]
    refined = _refine_pose(
        views,
        camera,
        prior,
        priors,
        xy[pnp_inliers],
        xyz[pnp_inliers],
        est.ransac.max_error,
        joint_prior_ba=options.joint_prior_ba,
    )
    if refined is None:
        counts["status"] = "insufficient_reference_geometry"
        return None, counts
    pose, counts["median_sampson_px"] = refined
    if counts["median_sampson_px"] > options.geometric_max_error:
        counts["status"] = "inconsistent_reference_geometry"
        return None, counts
    local = pose * xyz
    errors = np.linalg.norm(camera.img_from_cam(local) - xy, axis=1)
    mask = (local[:, 2] > 0) & (errors <= est.ransac.max_error)
    inliers = [edges[j] for j in np.flatnonzero(mask)]
    votes = Counter(n for e in inliers for n in sources[e])
    counts.update(inliers=len(inliers), reference_inliers=dict(votes))
    if (
        len(inliers) < mapper.abs_pose_min_num_inliers
        or mask.mean() < mapper.abs_pose_min_inlier_ratio
    ):
        counts["status"] = "insufficient_inliers"
        return None, counts
    if sum(n >= 6 for n in votes.values()) < 2:
        counts["status"] = "insufficient_reference_views"
        return None, counts
    delta = pose * prior.inverse()
    center_error = float(np.linalg.norm(delta.translation))
    angle = float(np.degrees(delta.rotation.angle()))
    counts.update(
        prior_center_difference=center_error,
        prior_angle_deg=angle,
        median_reprojection_px=float(np.median(errors[mask])),
    )
    if (
        center_error > 3 * priors.position_sigma
        or angle > 3 * priors.rotation_sigma_deg
    ):
        counts["status"] = "outside_prior_uncertainty"
        return None, counts
    counts["status"] = "registered"
    return (pose, keypoints, inliers), counts


def recover_prior_views(model, db_path, pairs, priors, options):
    """Register omitted views using the original reference map."""
    registered = {model.images[i].name: model.images[i] for i in model.reg_image_ids()}
    neighbors = defaultdict(list)
    for a, b in pairs:
        if a in registered and b not in registered:
            neighbors[b].append(registered[a])
        if b in registered and a not in registered:
            neighbors[a].append(registered[b])
    report = {}
    with pycolmap.Database.open(str(db_path)) as db:
        frames = {i.id: f for f in db.read_all_frames() for i in f.image_ids}
        for query in db.read_all_images():
            if query.name in registered:
                continue
            solution, report[query.name] = _localize(
                db, model, query, neighbors[query.name], priors, options
            )
            if solution is None:
                continue
            pose, keypoints, inliers = solution
            frame = frames[query.image_id]
            if not model.exists_camera(query.camera_id):
                model.add_camera(db.read_camera(query.camera_id))
            if not model.exists_rig(frame.rig_id):
                model.add_rig(db.read_rig(frame.rig_id))
            if not model.exists_frame(frame.frame_id):
                model.add_frame(frame)
            if not model.exists_image(query.image_id):
                query.frame_id = frame.frame_id
                model.add_image(query)
            image = model.images[query.image_id]
            if image.num_points2D() == 0:
                image.points2D = pycolmap.Image(keypoints=keypoints).points2D
            model.frames[frame.frame_id].set_cam_from_world(query.camera_id, pose)
            model.register_frame(frame.frame_id)
            for idx, pid in inliers:
                model.add_observation(pid, pycolmap.TrackElement(query.image_id, idx))
    if any(r["status"] == "registered" for r in report.values()):
        model.update_point_3d_errors()
    return report
