"""COLMAP image/camera import, feature storage and geometric verification."""

import numpy as np
import pycolmap

from .tracks import filter_verified_tracks


def create_database(path, images, names, sizes, store, options, camera_params, priors):
    if path.exists():
        raise FileExistsError(f"Refusing to replace an existing database: {path}")
    with pycolmap.Database.open(str(path)):
        pass
    if options.use_intrinsics_prior:
        groups = {}
        for name in names:
            groups.setdefault((*sizes[name], *priors.cameras[name]), []).append(name)
        for calibration, group in sorted(groups.items()):
            reader = pycolmap.ImageReaderOptions()
            reader.camera_model = "PINHOLE"
            reader.camera_params = ",".join(map(str, calibration[2:]))
            pycolmap.import_images(
                str(path),
                str(images),
                camera_mode=pycolmap.CameraMode.SINGLE,
                image_names=group,
                options=reader,
            )
    else:
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = options.camera_model or "SIMPLE_RADIAL"
        mode = (
            pycolmap.CameraMode.SINGLE
            if options.single_camera
            else pycolmap.CameraMode.AUTO
        )
        if camera_params is not None:
            reader.camera_model = "PINHOLE"
            reader.camera_params = ",".join(map(str, camera_params))
            mode = pycolmap.CameraMode.SINGLE
        pycolmap.import_images(
            str(path), str(images), camera_mode=mode, image_names=names, options=reader
        )
    with pycolmap.Database.open(str(path)) as db:
        ids = {im.name: im.image_id for im in db.read_all_images()}
        if set(ids) != set(names):
            raise RuntimeError(
                f"COLMAP imported unexpected images: {set(names) ^ set(ids)}"
            )
        with pycolmap.DatabaseTransaction(db):
            for name in names:
                # RoMaV2 already uses COLMAP's half-pixel origin.
                db.write_keypoints(ids[name], store.keypoints(name))
                if options.use_pose_prior:
                    db.write_pose_prior(
                        ids[name],
                        pycolmap.PosePrior(
                            priors.poses[name][:3, 3],
                            np.eye(3) * priors.position_sigma**2,
                            pycolmap.PosePriorCoordinateSystem.CARTESIAN,
                        ),
                    )
            for (a, b), matches in store.matches.items():
                db.write_matches(ids[a], ids[b], matches)
        cameras = {
            str(cam.camera_id): {
                "model": cam.model.name,
                "width": cam.width,
                "height": cam.height,
                "params": cam.params.tolist(),
            }
            for cam in db.read_all_cameras()
        }
    return ids, cameras


def _verify_prior_geometry(db, pairs, ids, priors, max_error):
    """Screen raw matches against the cameras that will actually triangulate them."""
    images = {im.name: im for im in db.read_all_images()}
    cameras = {cam.camera_id: cam for cam in db.read_all_cameras()}
    cam_from_world = {name: priors.cam_from_world(name) for name in ids}
    keypoints = {name: db.read_keypoints(i)[:, :2] for name, i in ids.items()}
    for a, b in pairs:
        matches = db.read_matches(ids[a], ids[b])
        geometry = pycolmap.TwoViewGeometry()
        if len(matches):
            cam_a, cam_b = (cameras[images[n].camera_id] for n in (a, b))
            rel = cam_from_world[b] * cam_from_world[a].inverse()
            xy_a = keypoints[a][matches[:, 0]]
            xy_b = keypoints[b][matches[:, 1]]
            rays_a = np.column_stack((cam_a.cam_from_img(xy_a), np.ones(len(matches))))
            rays_b = np.column_stack((cam_b.cam_from_img(xy_b), np.ones(len(matches))))
            geometry.cam2_from_cam1 = rel
            if np.linalg.norm(rel.translation) < 1e-9:
                # With identical centers only the rotation constrains a correspondence.
                pred_b = cam_b.img_from_cam(rays_a @ rel.rotation.matrix().T)
                pred_a = cam_a.img_from_cam(rays_b @ rel.rotation.matrix())
                valid = (np.linalg.norm(pred_a - xy_a, axis=1) <= max_error) & (
                    np.linalg.norm(pred_b - xy_b, axis=1) <= max_error
                )
                geometry.config = pycolmap.TwoViewGeometryConfiguration.PANORAMIC
            else:
                essential = pycolmap.essential_matrix_from_pose(rel)
                line_a, line_b = rays_b @ essential, rays_a @ essential.T
                numerator = np.abs(np.sum(line_b * rays_b, axis=1))
                # PINHOLE fx and fy can differ, so scale each normal per axis.
                normal_a = np.linalg.norm(line_a[:, :2] / cam_a.params[:2], axis=1)
                normal_b = np.linalg.norm(line_b[:, :2] / cam_b.params[:2], axis=1)
                normal = np.minimum(normal_a, normal_b)
                valid = (normal > 1e-15) & (numerator <= max_error * normal)
                # Cross products recover the depth signs in both cameras.
                ray_a_in_b = rays_a @ rel.rotation.matrix().T
                cross = np.cross(ray_a_in_b, rays_b)
                depth_a = np.sum(np.cross(rays_b, rel.translation) * cross, axis=1)
                depth_b = np.sum(np.cross(ray_a_in_b, rel.translation) * cross, axis=1)
                parallel = np.linalg.norm(cross, axis=1) <= (
                    1e-12
                    * np.linalg.norm(ray_a_in_b, axis=1)
                    * np.linalg.norm(rays_b, axis=1)
                )
                valid &= parallel | ((depth_a > 0) & (depth_b > 0))
                geometry.config = pycolmap.TwoViewGeometryConfiguration.CALIBRATED
                geometry.E = essential
            geometry.inlier_matches = matches[valid]
        db.write_two_view_geometry(ids[a], ids[b], geometry)


def verify_database(path, pairs_path, names, pairs, ids, store, options, priors=None):
    if not options.triangulate_from_priors:
        geometry = pycolmap.TwoViewGeometryOptions()
        geometry.min_num_inliers = options.min_matches
        geometry.ransac.random_seed = options.seed
        geometry.ransac.max_error = options.geometric_max_error
        geometry.ransac.max_num_trials = options.geometric_max_trials
        geometry.ransac.min_inlier_ratio = options.geometric_min_inlier_ratio
        pycolmap.verify_matches(str(path), str(pairs_path), geometry)
    with pycolmap.Database.open(str(path)) as db:
        if options.triangulate_from_priors:
            with pycolmap.DatabaseTransaction(db):
                _verify_prior_geometry(
                    db, pairs, ids, priors, options.geometric_max_error
                )
        verified_counts = {
            f"{a} {b}": len(db.read_two_view_geometry(ids[a], ids[b]).inlier_matches)
            for a, b in pairs
        }
        with pycolmap.DatabaseTransaction(db):
            filtered_counts, rejected = filter_verified_tracks(
                db, names, pairs, ids, store
            )
    return verified_counts, filtered_counts, rejected
