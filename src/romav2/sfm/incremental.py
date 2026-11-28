"""Grow a metric reconstruction from a verified pair and pose-only bundle adjustment."""

from collections import defaultdict

import numpy as np
import pycolmap
from scipy.optimize import least_squares
from scipy.spatial import Delaunay, QhullError, cKDTree
from scipy.spatial.transform import Rotation

from .io import write_json


def _map_observations(model, query_id, references, links):
    from .registration import _correspondences

    edges, _ = _correspondences(
        (model.image(i), links[i][query_id]) for i in references
    )
    query = model.image(query_id)
    return [(pid, k, query.point2D(k).xy) for k, pid in sorted(edges)]


def _transfer_pixels(source, target, xy):
    mesh = Delaunay(source)
    simplex = mesh.find_simplex(xy)
    vertices = mesh.simplices[simplex]
    transform = mesh.transform[simplex]
    first = np.einsum("nij,nj->ni", transform[:, :2], xy - transform[:, 2])
    weights = np.column_stack((first, 1 - first.sum(axis=1)))
    transferred = (weights[:, :, None] * target[vertices]).sum(axis=1)
    distances = np.linalg.norm(source[vertices] - xy[:, None], axis=2)
    valid = (
        (simplex >= 0)
        & (distances.max(axis=1) <= 40)
        & np.isfinite(transferred).all(axis=1)
    )
    return transferred, valid


def _transfer_landmarks(model, query_id, references, pixels, links, direct):
    # Independently sampled RoMa pairs rarely reuse all landmark pixels. Interpolate
    # the verified local warp in both directions. Only database pixels can be attached.
    observations = {pid: (pid, k, xy) for pid, k, xy in direct}
    occupied = {k for _, k, _ in direct}
    tree = cKDTree(pixels[query_id])
    for i in references[:3]:
        ref = model.image(i)
        idx = np.asarray(ref.get_observation_point2D_idxs(), dtype=int)
        matches = links[i][query_id]
        if not len(idx) or len(matches) < 4:
            continue
        source, target = pixels[i][matches[:, 0]], pixels[query_id][matches[:, 1]]
        try:
            dest, inside = _transfer_pixels(source, target, pixels[i][idx])
            dest, idx = dest[inside], idx[inside]
            if not len(idx):
                continue
            back, valid = _transfer_pixels(target, source, dest)
        except QhullError:
            continue
        valid &= np.linalg.norm(back - pixels[i][idx], axis=1) < 1.0
        distance, keys = tree.query(dest[valid])
        for j, uv, d, key in zip(idx[valid], dest[valid], distance, keys):
            pid = ref.point2D(int(j)).point3D_id
            if pid in observations:
                continue
            k = int(key) if d <= 1 and key not in occupied else None
            if k is not None:
                occupied.add(k)
            observations[pid] = (pid, k, pixels[query_id][k] if k is not None else uv)
    return list(observations.values())


def _localize(model, query, edges, start, priors, options):
    limit = options.geometric_max_error
    report = dict(candidates=len(edges), status="insufficient_tracks")
    if len(edges) < 16:
        return None, report
    xy = np.stack([pixel for _, _, pixel in edges])
    xyz = np.stack([model.point3D(p).xyz for p, _, _ in edges])
    # Hold out whole image tiles so nearby dense samples do not straddle the split.
    tiles = np.floor(xy / 48).astype(int)
    _, groups = np.unique(tiles, axis=0, return_inverse=True)
    rng = np.random.default_rng((options.seed + query.image_id) % 2**32)
    train = ~np.isin(groups, rng.permutation(groups.max() + 1)[::4])
    if train.sum() < 8 or (~train).sum() < 3:
        return None, {**report, "status": "poor_image_coverage"}
    est = pycolmap.AbsolutePoseEstimationOptions()
    est.ransac.max_error = limit
    est.ransac.random_seed = options.seed
    pnp = pycolmap.estimate_absolute_pose(xy[train], xyz[train], query.camera, est)
    focal, principal = np.split(query.camera.params, 2)
    sigma_r = np.deg2rad(priors.rotation_sigma_deg)
    center = start.inverse().translation
    rotation = start.rotation.matrix()

    def step_from(pose):
        return np.r_[
            Rotation.from_matrix(pose.rotation.matrix() @ rotation.T).as_rotvec()
            / sigma_r,
            (pose.inverse().translation - center) / priors.position_sigma,
        ]

    def project(step, points):
        r = Rotation.from_rotvec(step[:3] * sigma_r).as_matrix() @ rotation
        c = center + step[3:] * priors.position_sigma
        local = (points - c) @ r.T
        uv = local[:, :2] / np.maximum(local[:, 2:], 1e-8) * focal + principal
        return uv, local[:, 2]

    def residual(step, support):
        uv, _ = project(step, xyz[support])
        delta = uv - xy[support]
        weight = np.sqrt(2 / (1 + np.sqrt(1 + np.sum(delta**2, axis=1))))
        return np.r_[
            (delta * weight[:, None]).ravel() * np.sqrt(min(1.0, 2 / len(support))),
            step,
        ]

    def errors(step):
        uv, depth = project(step, xyz)
        return np.where(depth > 0, np.linalg.norm(uv - xy, axis=1), np.inf)

    zero = np.zeros(6)

    def credible(step):
        return all(np.linalg.norm(part) <= 4 for part in np.split(step, 2))

    prior_errors = errors(zero)
    candidates = [
        ("initial_pose", zero, np.flatnonzero(train & (prior_errors <= 2 * limit)))
    ]
    if pnp is not None and pnp["num_inliers"] >= 8:
        proposal = step_from(pnp["cam_from_world"])
        if credible(proposal):
            candidates.append(
                ("pnp", proposal, np.flatnonzero(train)[pnp["inlier_mask"]])
            )

    before = float(np.minimum(prior_errors[~train], limit).mean())
    minimum_held = max(3, (~train).sum() * options.geometric_min_inlier_ratio)
    attachable = np.array([idx is not None for _, idx, _ in edges])
    best = None
    trials = {}
    for name, initial, support in candidates:
        if len(support) < 8:
            continue
        fit = least_squares(
            residual,
            initial,
            args=(support,),
            max_nfev=options.registration_iterations + 1,
        )
        step = fit.x
        if not np.isfinite(step).all():
            trials[name] = dict(status="nonfinite_pose", evaluations=fit.nfev)
            continue
        err = errors(step)
        held = err[~train]
        held_after = float(np.minimum(held, limit).mean())
        inliers = err <= limit
        details = dict(
            center_update_m=float(np.linalg.norm(step[3:]) * priors.position_sigma),
            rotation_update_deg=float(
                np.linalg.norm(step[:3]) * priors.rotation_sigma_deg
            ),
            attachable_inliers=int((inliers & attachable).sum()),
            held_out=len(held),
            held_inliers=int((held <= limit).sum()),
            held_before_px=before,
            held_after_px=held_after,
            inliers=int(inliers.sum()),
            evaluations=fit.nfev,
            status="pose_validation_failed",
        )
        trials[name] = details
        if (
            not credible(step)
            or details["attachable_inliers"] < 3
            or details["inliers"] < 12
            or details["held_inliers"] < minimum_held
            or held_after > before + 0.25
        ):
            continue
        uv = xy[inliers]
        spread = np.linalg.svd(uv - uv.mean(axis=0), compute_uv=False)
        if spread[-1] / np.sqrt(len(uv)) < limit:
            details["status"] = "poor_image_coverage"
            continue
        # Use the same robust visual weight and standardized pose penalty as the fit.
        cost = 2 * np.mean(np.hypot(1, np.minimum(held, limit)) - 1) + 0.5 * (
            step @ step
        )
        details.update(status="accepted", validation_cost=float(cost))
        if best is None or cost < best[0]:
            best = cost, name, initial, step, inliers

    report["trials"] = trials
    if best is None:
        if trials:
            report.update(
                min(
                    trials.values(),
                    key=lambda trial: trial.get("held_after_px", np.inf),
                )
            )
        else:
            report["status"] = "insufficient_inliers"
        return None, report

    _, name, initial, step, inliers = best

    def pose_at(delta):
        r = Rotation.from_rotvec(delta[:3] * sigma_r).as_matrix() @ rotation
        c = center + delta[3:] * priors.position_sigma
        return pycolmap.Rigid3d(pycolmap.Rotation3d(r), -r @ c)

    report.update(trials[name])
    report.update(
        status="registered",
        selected_start=name,
        ba_start_camera_to_world=pose_at(initial).inverse().matrix().tolist(),
    )
    return (pose_at(step), [edges[i] for i in np.flatnonzero(inliers)]), report


def _rank_seeds(links, initial, names, cache, pixels, min_angle, preferred):
    seeds = []
    for a in links:
        for b, m in links[a].items():
            if a >= b:
                continue
            # A seed needs repeat observations in a third image to grow.
            support = defaultdict(set)
            for i, j, idx in ((a, b, 0), (b, a, 1)):
                for q, mq in links[i].items():
                    if q != j:
                        support[q].update(mq[np.isin(mq[:, 0], m[:, idx]), 1].tolist())
            best = max(map(len, support.values()), default=0)

            poses = [initial.cam_from_world(names[i]) for i in (a, b)]
            rays = []
            for i, idx, pose in zip((a, b), m.T, poses):
                camera = cache.cameras[cache.images[i].camera_id]
                ray = (
                    np.c_[camera.cam_from_img(pixels[i][idx]), np.ones(len(idx))]
                    @ pose.rotation.matrix()
                )
                rays.append(ray / np.linalg.norm(ray, axis=1)[:, None])
            angle = np.degrees(
                np.arctan2(
                    np.linalg.norm(np.cross(*rays), axis=1),
                    np.sum(rays[0] * rays[1], axis=1),
                )
            )
            good = int(((angle >= min_angle) & (angle < 90)).sum())
            if good >= 16:
                seeds.append(
                    (
                        set((names[a], names[b])) == set(preferred or ()),
                        best * good / len(m),
                        good,
                        a,
                        b,
                    )
                )
    return sorted(seeds, reverse=True)


def _register_pose(model, image_id, pose):
    image = model.image(image_id)
    model.frame(image.frame_id).set_cam_from_world(image.camera_id, pose)
    model.register_frame(image.frame_id)


def _grow(
    model,
    manager,
    triangulator,
    tri,
    seed_pair,
    links,
    pixels,
    names,
    initial,
    priors,
    options,
):
    order = list(seed_pair)
    details = {}
    while True:
        remaining = set(links).difference(order)
        ranked = [
            (sum(len(links[q][i]) for i in order if i in links[q]), q)
            for q in remaining
        ]
        for strength, q in sorted(ranked, reverse=True):
            if not strength:
                continue
            refs = [i for i in order if i in links[q]]
            refs.sort(key=lambda i: len(links[i][q]), reverse=True)
            query = model.image(q)
            start = initial.cam_from_world(query.name)
            edges = _map_observations(model, q, refs, links)
            solution, detail = _localize(model, query, edges, start, priors, options)
            detail["source"] = "verified_tracks"
            if solution is None:
                edges = _transfer_landmarks(model, q, refs, pixels, links, edges)
                solution, detail = _localize(
                    model, query, edges, start, priors, options
                )
                detail["source"] = "verified_tracks_and_local_warp"
            detail["references"] = [names[i] for i in refs]
            details[query.name] = detail
            if solution is None:
                continue
            pose, inliers = solution
            detail.update(
                initial_camera_to_world=start.inverse().matrix().tolist(),
                refined_camera_to_world=pose.inverse().matrix().tolist(),
            )
            _register_pose(model, q, pose)
            for pid, idx, _ in inliers:
                if idx is None:
                    continue
                manager.add_observation(pid, pycolmap.TrackElement(q, idx))
            triangulator.triangulate_image(tri, q)
            changed = {
                p.point3D_id for p in model.images[q].points2D if p.has_point3D()
            }
            manager.filter_points3D(options.geometric_max_error, tri.min_angle, changed)
            order.append(q)
            break
        else:
            break
    return order, details


def incremental_from_priors(
    db_path, images, sparse, initial, priors, options, preferred=None
):
    with pycolmap.Database.open(str(db_path)) as db:
        cache = pycolmap.DatabaseCache.create(db, options.min_matches, False, set())
        names = {im.image_id: im.name for im in db.read_all_images()}
        pixels = {i: db.read_keypoints(i)[:, :2] for i in names}

    graph = cache.correspondence_graph
    links = defaultdict(dict)
    for key in graph.num_correspondences_between_all_images():
        a, b = pycolmap.pair_id_to_image_pair(key)
        matches = graph.find_correspondences_between_images(a, b)
        links[a][b] = matches
        links[b][a] = matches[:, ::-1]

    tri = pycolmap.IncrementalTriangulatorOptions()
    tri.ignore_two_view_tracks = False
    tri.random_seed = options.seed
    seeds = _rank_seeds(links, initial, names, cache, pixels, tri.min_angle, preferred)
    attempts = []
    report = dict(
        units="meters",
        seed_attempts=attempts,
        iteration_limit=options.registration_iterations,
    )
    path = sparse.parent / "incremental_registration.json"
    best_model = None
    best_order = []
    best_details = {}
    while seeds:
        _, _, _, a, b = seeds.pop(0)
        model = pycolmap.Reconstruction()
        model.load(cache)
        manager = pycolmap.ObservationManager(model, graph)
        triangulator = pycolmap.IncrementalTriangulator(graph, model, manager)

        for i in (a, b):
            _register_pose(model, i, initial.cam_from_world(names[i]))
        triangulator.triangulate_image(tri, b)
        manager.filter_all_points3D(options.geometric_max_error, tri.min_angle)
        attempts.append(
            dict(
                pair=[names[a], names[b]],
                seed_points=model.num_points3D(),
                baseline_m=float(
                    np.linalg.norm(
                        model.image(a).projection_center()
                        - model.image(b).projection_center()
                    )
                ),
            )
        )
        if model.num_points3D() < 16:
            continue
        order, details = _grow(
            model,
            manager,
            triangulator,
            tri,
            (a, b),
            links,
            pixels,
            names,
            initial,
            priors,
            options,
        )
        attempts[-1]["registered_images"] = len(order)
        if len(order) < 3:
            continue
        if len(order) > len(best_order):
            best_model, best_order, best_details = model, order, details
        if (
            len(order) == len(names)
            or sum(x.get("registered_images", 0) >= 3 for x in attempts) >= 3
        ):
            break
        # Restart across the boundary of a stalled map. Each attempt grows one
        # connected reconstruction and competes on verified coverage.
        registered = set(best_order)
        seeds = [seed for seed in seeds if not set(seed[-2:]) <= registered]
        seeds.sort(
            key=lambda seed: (len(registered.intersection(seed[-2:])) == 1, seed),
            reverse=True,
        )
    if best_model is not None:
        model, order, details = best_model, best_order, best_details
        a, b = order[:2]
        report.update(
            order=[names[i] for i in order],
            frames={
                n: details.get(n, {"status": "insufficient_tracks"})
                for i, n in names.items()
                if i not in (a, b)
            },
        )
        write_json(path, report)
        model.extract_colors_for_all_images(str(images))
        model.update_point_3d_errors()
        (sparse / "0").mkdir(exist_ok=True)
        return {0: model}
    write_json(path, report)
    raise RuntimeError(f"No seed supported a validated third view; see {path}")
