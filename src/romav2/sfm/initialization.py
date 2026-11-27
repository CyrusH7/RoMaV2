"""Prior-guided refinement of a calibrated initial image pair."""

from collections import defaultdict
from dataclasses import replace

import numpy as np
import pycolmap
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .seed_landmarks import refine_landmark_pair, sampson_pixels as _sampson_pixels

MIN_TRIANGULATION_SIN = np.sin(np.deg2rad(1.5))


def _triangulation_support(rotation, direction, rays_a, rays_b):
    rotated = rays_a @ rotation.T
    cross = np.cross(rotated, rays_b)
    depth_a = np.sum(np.cross(rays_b, direction) * cross, axis=1)
    depth_b = np.sum(np.cross(rotated, direction) * cross, axis=1)
    sin_angle = np.linalg.norm(cross, axis=1) / (
        np.linalg.norm(rotated, axis=1) * np.linalg.norm(rays_b, axis=1)
    )
    return (depth_a > 0) & (depth_b > 0), sin_angle


def _score_pair(relative, observations, threshold):
    length = np.linalg.norm(relative.translation)
    if length <= 1e-9:
        return 1.0, 0, 0.0
    rotation = relative.rotation.matrix()
    direction = relative.translation / length
    errors = np.abs(_sampson_pixels(rotation, direction, *observations))
    positive, sin_angle = _triangulation_support(rotation, direction, *observations[:2])
    inliers = positive & (errors <= threshold) & (sin_angle >= MIN_TRIANGULATION_SIN)
    loss = np.where(inliers, errors / threshold, 1.0)
    return float(loss.mean()), int(inliers.sum()), float(np.median(errors))


def _fit_rotation(observations, threshold, relative_pose, reference_residuals):
    def residual(step):
        relative = relative_pose(step)
        error = _sampson_pixels(
            relative.rotation.matrix(), relative.translation, *observations
        )
        magnitude = np.abs(error)
        huber = np.sign(error) * np.sqrt(
            np.where(
                magnitude <= threshold,
                error**2,
                2 * threshold * magnitude - threshold**2,
            )
        )
        return np.r_[huber / np.sqrt(len(error)), reference_residuals(relative), step]

    result = least_squares(residual, np.zeros(3), method="lm", max_nfev=80)
    return (
        relative_pose(result.x)
        if result.success and np.isfinite(result.x).all()
        else None
    )


def _fit_relative_motion(prior, observations, threshold, sigma_rotation, sigma_center):
    """Fit relative rotation and translation direction at the input metric scale."""
    baseline = np.linalg.norm(prior.translation)
    direction = prior.translation / baseline
    # Two orthonormal tangent directions on the unit sphere.
    axis = np.eye(3)[np.argmin(np.abs(direction))]
    u = np.cross(direction, axis)
    u /= np.linalg.norm(u)
    basis = np.column_stack((u, np.cross(direction, u)))
    sigma_direction = min(1.0, sigma_center / baseline)
    r0 = prior.rotation.matrix()

    def unpack(step):
        r = Rotation.from_rotvec(step[:3] * sigma_rotation).as_matrix() @ r0
        d = direction + basis @ (sigma_direction * step[3:])
        return r, d / np.linalg.norm(d)

    step = np.zeros(5)
    for gate in (4 * threshold, 2 * threshold, threshold):
        err = _sampson_pixels(*unpack(step), *observations)
        keep = np.abs(err) <= gate
        if keep.sum() < 8:
            return None
        obs = (observations[0][keep], observations[1][keep], *observations[2:])
        weights = np.sqrt(
            np.minimum(
                1.0, max(1.0, threshold / 2) / np.maximum(np.abs(err[keep]), 1e-12)
            )
            / keep.sum()
        )

        def residual(delta):
            return np.r_[weights * _sampson_pixels(*unpack(delta), *obs), 0.25 * delta]

        fit = least_squares(residual, step, method="lm", max_nfev=80)
        if not fit.success or not np.isfinite(fit.x).all():
            return None
        step = fit.x
    r, d = unpack(step)
    return pycolmap.Rigid3d(pycolmap.Rotation3d(r), baseline * d)


def _metric_baseline(
    anchor, relative, rays, tracks, sigma_center, sigma_rotation, min_support
):
    # In a third camera a seed point has coordinates offset + baseline * ray.
    # Each independent camera votes once, regardless of how many tracks it sees.
    length = np.linalg.norm(relative.translation)
    d = relative.translation / length
    r = relative.rotation.matrix()
    x, y = rays
    rotated = x @ r.T
    cross = np.cross(rotated, y)
    depth = np.sum(np.cross(y, d) * cross, axis=1) / np.maximum(
        np.sum(cross**2, axis=1), 1e-15
    )
    positive, angle = _triangulation_support(r, d, x, y)
    valid = positive & (angle >= MIN_TRIANGULATION_SIN)
    votes = []
    uncertainties = []
    for camera, indices, uv, _ in tracks:
        transform = camera * anchor.inverse()
        ray = (x[indices] * depth[indices, None]) @ transform.rotation.matrix().T
        h = ray[:, :2] - uv * ray[:, 2:]
        k = transform.translation[:2] - uv * transform.translation[2]
        norm_sq = np.sum(h**2, axis=1)
        estimates = -np.sum(h * k, axis=1) / np.maximum(norm_sq, 1e-15)
        good = (
            valid[indices]
            & (norm_sq > MIN_TRIANGULATION_SIN**2 * np.sum(ray**2, axis=1))
            & (estimates > 0)
            & (transform.translation[2] + estimates * ray[:, 2] > 0)
        )
        if good.sum() >= min_support:
            votes.append(np.median(estimates[good]))
            # Propagate anchor and reference pose uncertainty through the
            # scalar scale fit. Pose errors are shared by tracks in a view.
            normal = np.column_stack((h[good], -np.sum(h[good] * uv[good], axis=1)))
            point = estimates[good, None] * ray[good] + transform.translation
            translation_var = sigma_center**2 * np.sum(normal**2, axis=1)
            rotation_var = sigma_rotation**2 * np.sum(
                np.cross(point, normal) ** 2, axis=1
            )
            uncertainties.append(
                np.median(np.sqrt(2 * (translation_var + rotation_var)) / norm_sq[good])
            )
    if len(votes) >= 3:
        estimate = float(np.median(votes))
        scatter = 1.4826 * np.median(np.abs(np.asarray(votes) - estimate))
        uncertainty = max(float(np.median(uncertainties)), float(scatter))
        correction = estimate - length
        if 2 * uncertainty < abs(correction) <= 2 * sigma_center:
            weight = sigma_center**2 / (sigma_center**2 + uncertainty**2)
            return float(length + weight * correction), len(votes)
    return float(length), len(votes)


def _third_view_error(relative, world_from_a, stereo_rays, tracks, threshold):
    rays_a, rays_b = stereo_rays
    rotation, translation = relative.rotation.matrix(), relative.translation
    rotated = rays_a @ rotation.T
    cross = np.cross(rotated, rays_b)
    denominator = np.sum(cross**2, axis=1)
    depth_a = np.sum(np.cross(rays_b, translation) * cross, axis=1)
    depth_b = np.sum(np.cross(rotated, translation) * cross, axis=1)
    valid = (depth_a > 0) & (depth_b > 0)
    depth = depth_a / np.maximum(denominator, 1e-15)
    xyz_world = world_from_a * (rays_a * depth[:, None])
    residuals = []
    for camera_from_world, indices, xy_normalized, focal in tracks:
        observed = camera_from_world * xyz_world[indices]
        positive = valid[indices] & (observed[:, 2] > 0)
        projected = observed[:, :2] / np.maximum(observed[:, 2:], 1e-8)
        error = np.linalg.norm((projected - xy_normalized) * focal, axis=1)
        residuals.append(np.where(positive, np.minimum(error, threshold), threshold))
    return float(np.concatenate(residuals).mean())


def _shared_third_view_tracks(
    a, b, held_matches, matches_by_pair, neighbors, cameras, rays, poses
):
    def links(source, target):
        edge = tuple(sorted((source, target)))
        matches = matches_by_pair[edge]
        return dict(matches if source == edge[0] else matches[:, ::-1])

    linked_a = {v if u == a else u for u, v in neighbors[a]}
    linked_b = {v if u == b else u for u, v in neighbors[b]}
    tracks = []
    for view in sorted(linked_a & linked_b):
        from_a, from_b = links(a, view), links(b, view)
        indices, observations = [], []
        for i, (ia, ib) in enumerate(held_matches):
            target = from_a.get(ia)
            if target is not None and target == from_b.get(ib):
                indices.append(i)
                observations.append(target)
        if indices:
            tracks.append(
                (
                    poses[view],
                    indices,
                    rays[view][observations, :2],
                    cameras[view].params[:2],
                )
            )
    return tracks


def refine_initial_pair(db_path, pairs, ids, priors, options, *, seed_pair=None):
    """Refine one initial pair, optionally restricting the candidate search."""
    joint = options.refine_initial_pose == "joint"
    poses = {name: priors.cam_from_world(name) for name in ids}
    threshold = options.geometric_max_error
    min_support = max(20, options.min_matches)
    min_baseline = priors.position_sigma
    sigma_rotation = np.deg2rad(priors.rotation_sigma_deg)
    neighbors = defaultdict(list)
    matches_by_pair = {}
    ranking = []
    with pycolmap.Database.open(str(db_path)) as db:
        cameras = {cam.camera_id: cam for cam in db.read_all_cameras()}
        image_cameras = {im.name: cameras[im.camera_id] for im in db.read_all_images()}
        rays = {
            name: np.column_stack(
                (
                    camera.cam_from_img(db.read_keypoints(ids[name])[:, :2]),
                    np.ones(db.num_keypoints_for_image(ids[name])),
                )
            )
            for name, camera in image_cameras.items()
        }
        for a, b in pairs:
            matches = db.read_matches(ids[a], ids[b])
            if len(matches) >= min_support:
                matches_by_pair[a, b] = matches
                neighbors[a].append((a, b))
                neighbors[b].append((a, b))

    def observations(a, b):
        matches = matches_by_pair[a, b]
        return (
            rays[a][matches[:, 0]],
            rays[b][matches[:, 1]],
            image_cameras[a].params[:2],
            image_cameras[b].params[:2],
        )

    def held_out_indices(count):
        rng = np.random.default_rng(options.seed % 2**32)
        return rng.permutation(count)[::4]

    def split(obs):
        test = held_out_indices(len(obs[0]))
        train = np.ones(len(obs[0]), dtype=bool)
        train[test] = False
        return (obs[0][train], obs[1][train], *obs[2:]), (
            obs[0][test],
            obs[1][test],
            *obs[2:],
        )

    for a, b in matches_by_pair:
        if seed_pair is not None and (a, b) != seed_pair:
            continue
        initial = poses[b] * poses[a].inverse()
        if np.linalg.norm(initial.translation) < min_baseline:
            continue
        _, count, _ = _score_pair(initial, observations(a, b), 2 * threshold)
        if count >= min_support:
            ranking.append((count, a, b))

    report = {
        "status": "kept_input_poses",
        "min_baseline_m": min_baseline,
        "candidates": [],
        "translation_locked": not joint,
        "mode": "joint" if joint else "rotation",
    }
    if options.refine_initial_pose == "landmarks":
        report.update(mode="landmarks", translation_locked=False)
        best = None
        for _, a, b in sorted(ranking, key=lambda item: -item[0])[:12]:
            matches = matches_by_pair[a, b]
            tracks = _shared_third_view_tracks(
                a, b, matches, matches_by_pair, neighbors, image_cameras, rays, poses
            )
            matrices, detail = refine_landmark_pair(
                (poses[a], poses[b]),
                observations(a, b),
                tracks,
                held_out_indices(len(matches)),
                sigma_rotation,
                priors.position_sigma,
                threshold,
                min_support,
            )
            detail.update(
                pair=[a, b],
                status="accepted" if matrices is not None else "kept_input_poses",
            )
            report["candidates"].append(detail)
            if matrices is not None:
                gain = detail["transfer_before_px"] - detail["transfer_after_px"]
                if best is None or gain > best[0]:
                    best = gain, (a, b), matrices, detail
        if best is None:
            return priors, report
        _, pair, matrices, detail = best
        updated = dict(zip(pair, matrices))
        detail["camera_to_world_by_image"] = {n: m.tolist() for n, m in updated.items()}
        report.update(status="refined", selected=detail)
        return replace(priors, poses={**priors.poses, **updated}), report

    selected = None
    scores = {}
    for _, a, b in sorted(ranking, key=lambda item: -item[0])[:12]:
        fit_obs, held_obs = split(observations(a, b))
        seed_matches = matches_by_pair[a, b]
        held_matches = seed_matches[held_out_indices(len(seed_matches))]
        third_views = _shared_third_view_tracks(
            a, b, held_matches, matches_by_pair, neighbors, image_cameras, rays, poses
        )
        num_third_view_observations = sum(len(track[1]) for track in third_views)
        prior = poses[b] * poses[a].inverse()
        before = _score_pair(prior, held_obs, threshold)
        candidate = {
            "pair": [a, b],
            "baseline_m": float(np.linalg.norm(prior.translation)),
            "held_out_matches": len(held_obs[0]),
            "held_out_score_before": before[0],
            "held_out_median_px_before": before[2],
            "held_out_triangulatable_before": before[1],
            "status": "insufficient_validation_tracks",
        }
        report["candidates"].append(candidate)
        if joint and before[2] <= min(1.0, threshold):
            candidate["status"] = "prior_within_pixel_noise"
            continue
        distinct_third_view_tracks = len(
            {idx for _, indices, _, _ in third_views for idx in indices}
        )
        if distinct_third_view_tracks < min_support:
            continue
        third_before = _third_view_error(
            prior, poses[a].inverse(), held_obs[:2], third_views, 2 * threshold
        )

        if joint:
            motion = _fit_relative_motion(
                prior, fit_obs, threshold, sigma_rotation, priors.position_sigma
            )
            if motion is None:
                candidate["status"] = "joint_fit_failed"
                continue
            train_mask = np.ones(len(seed_matches), dtype=bool)
            train_mask[held_out_indices(len(seed_matches))] = False
            scale_tracks = _shared_third_view_tracks(
                a,
                b,
                seed_matches[train_mask],
                matches_by_pair,
                neighbors,
                image_cameras,
                rays,
                poses,
            )
        candidate["status"] = "insufficient_neighbor_matches"
        alternatives = []
        for name, anchor in ((b, a), (a, b)):
            incident, support = [], []
            held_keypoints = held_matches[:, 1] if name == b else held_matches[:, 0]
            for u, v in neighbors[name]:
                if (u, v) == (a, b):
                    continue
                initial = poses[v] * poses[u].inverse()
                obs = observations(u, v)
                neighbor_fit, neighbor_test = split(obs)
                neighbor_matches = matches_by_pair[u, v]
                test_indices = held_out_indices(len(neighbor_matches))
                training = np.ones(len(neighbor_matches), dtype=bool)
                training[test_indices] = False
                own_indices = neighbor_matches[training, int(name == v)]
                if (u, v) not in scores:
                    scores[u, v] = _score_pair(initial, neighbor_test, threshold)[0]
                incident.append((u, v, neighbor_test))
                if np.linalg.norm(initial.translation) < min_baseline:
                    continue
                errors = np.abs(
                    _sampson_pixels(
                        initial.rotation.matrix(), initial.translation, *neighbor_fit
                    )
                )
                reliable = (errors <= 2 * threshold) & ~np.isin(
                    own_indices, held_keypoints
                )
                if reliable.sum() >= min_support:
                    support.append(
                        (
                            u,
                            v,
                            (
                                neighbor_fit[0][reliable],
                                neighbor_fit[1][reliable],
                                *neighbor_fit[2:],
                            ),
                        )
                    )

            def absolute(relative):
                delta = relative if name == b else relative.inverse()
                return delta * poses[anchor]

            def neighbor_pose(refined, u, v):
                view_u = refined if u == name else poses[u]
                view_v = refined if v == name else poses[v]
                return view_v * view_u.inverse()

            def reference_residuals(relative):
                refined = absolute(relative)
                residuals = []
                for u, v, obs in support:
                    relative = neighbor_pose(refined, u, v)
                    errors = _sampson_pixels(
                        relative.rotation.matrix(), relative.translation, *obs
                    )
                    magnitude = np.abs(errors)
                    squared = np.where(magnitude <= 2, errors**2, 4 * magnitude - 4)
                    huber = np.sign(errors) * np.sqrt(squared)
                    residuals.append(huber / np.sqrt(len(errors) * len(support)))
                return np.concatenate(residuals)

            if not support or (joint and len(support) < 2):
                continue
            candidate["status"] = "held_out_geometry_rejected"
            center = poses[name].inverse().translation
            orientation = poses[name].rotation.matrix()

            def relative_pose(step):
                rotation = (
                    Rotation.from_rotvec(step[:3] * sigma_rotation).as_matrix()
                    @ orientation
                )
                refined = pycolmap.Rigid3d(
                    pycolmap.Rotation3d(rotation), -rotation @ center
                )
                return neighbor_pose(refined, a, b)

            scale_views = 0
            if joint:
                from_anchor = motion if anchor == a else motion.inverse()
                anchor_rays = fit_obs[:2] if anchor == a else (fit_obs[1], fit_obs[0])
                length, scale_views = _metric_baseline(
                    poses[anchor],
                    from_anchor,
                    anchor_rays,
                    scale_tracks,
                    priors.position_sigma,
                    sigma_rotation,
                    min_support,
                )
                constrained = pycolmap.Rigid3d(
                    motion.rotation,
                    motion.translation * (length / candidate["baseline_m"]),
                )
            else:
                constrained = _fit_rotation(
                    fit_obs, threshold, relative_pose, reference_residuals
                )
            if constrained is None:
                continue
            if joint and (
                constrained.translation @ prior.translation <= 0
                or np.linalg.norm(absolute(constrained).inverse().translation - center)
                > 2 * priors.position_sigma
            ):
                continue
            after = _score_pair(constrained, held_obs, threshold)
            if after[0] >= before[0] or after[1] < max(5, before[1]):
                continue
            refined = absolute(constrained)
            pose_a = refined if name == a else poses[a]
            third_after = _third_view_error(
                constrained,
                pose_a.inverse(),
                held_obs[:2],
                third_views,
                2 * threshold,
            )
            if third_after >= third_before:
                continue
            shift = refined * poses[name].inverse()
            if shift.rotation.angle() > 2 * sigma_rotation:
                continue
            changes = [
                scores[u, v]
                - _score_pair(neighbor_pose(refined, u, v), obs, threshold)[0]
                for u, v, obs in incident
            ]
            if changes and np.mean(changes) < -0.01:
                continue
            gain = (before[0] - after[0] + sum(changes)) / (1 + len(changes))
            if gain > 0:
                details = dict(
                    status="accepted",
                    refined_image=name,
                    neighbors=len(incident),
                    baseline_support_views=scale_views,
                    support_views=len(support),
                    third_view_observations=num_third_view_observations,
                    third_view_tracks=distinct_third_view_tracks,
                    third_view_clipped_px_before=third_before,
                    third_view_clipped_px_after=third_after,
                    support_gain=gain,
                    refined_baseline_m=float(np.linalg.norm(constrained.translation)),
                    relative_t_change_m=float(
                        np.linalg.norm(constrained.translation - prior.translation)
                    ),
                    relative_t_direction_shift_deg=float(
                        np.rad2deg(
                            np.arctan2(
                                np.linalg.norm(
                                    np.cross(prior.translation, constrained.translation)
                                ),
                                prior.translation @ constrained.translation,
                            )
                        )
                    ),
                    held_out_median_px_after=after[2],
                    held_out_score_after=after[0],
                    held_out_triangulatable_after=after[1],
                    center_shift_m=float(
                        np.linalg.norm(refined.inverse().translation - center)
                    ),
                    rotation_shift_deg=float(np.degrees(shift.rotation.angle())),
                    camera_to_world=_camera_to_world(refined).tolist(),
                )
                alternatives.append((gain, name, details))
        if not alternatives:
            continue
        gain, name, details = max(alternatives, key=lambda item: item[0])
        candidate.update(details)
        if selected is None or gain > selected[0]:
            selected = gain, name, candidate

    if selected is None:
        return priors, report
    _, name, candidate = selected
    updated = priors.poses.copy()
    updated[name] = np.asarray(candidate["camera_to_world"])
    report.update(
        status="refined",
        selected=candidate,
        camera_to_world=updated[name].tolist(),
    )
    return replace(priors, poses=updated), report


def _camera_to_world(pose):
    matrix = np.eye(4)
    matrix[:3] = pose.inverse().matrix()
    return matrix
