"""Validated OpenCV camera priors and alternating pose/structure refinement.

All poses are camera-to-world, with x right, y down, z forward. Translations
and position sigmas use the same metric unit (meters in the quick experiment).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .calibration import validate_camera_to_world


@dataclass
class Priors:
    cameras: dict
    poses: dict
    position_sigma: float = 0.05
    rotation_sigma_deg: float = 3.0
    calibration: dict | None = None
    rig_calibration: dict | None = None

    def cam_from_world(self, name):
        """Convert the input camera-to-world prior to COLMAP's pose convention."""
        return pycolmap.Rigid3d(self.poses[name][:3]).inverse()


def load_priors(path, names):
    data = json.loads(Path(path).read_text())
    if data.get("pose_convention") != "camera_to_world_opencv":
        raise ValueError(
            "prior JSON must declare pose_convention=camera_to_world_opencv"
        )
    if data.get("units") != "meters":
        raise ValueError("prior JSON must declare units=meters")
    if data.get("pose_sensor") == "depth":
        raise ValueError("Convert depth camera poses to RGB before using RGB images")
    cameras, poses = {}, {}
    for name in names:
        entry = data.get("images", {}).get(name)
        if entry is None:
            raise ValueError(f"Missing prior entry for {name}")
        if "intrinsics" in entry:
            k = np.asarray(entry["intrinsics"], dtype=float)
            if k.shape != (4,) or not np.isfinite(k).all() or np.any(k[:2] <= 0):
                raise ValueError(f"Invalid fx fy cx cy for {name}")
            cameras[name] = k
        if "camera_to_world" in entry:
            poses[name] = validate_camera_to_world(entry["camera_to_world"], name)
    position_sigma = float(data.get("position_sigma", 0.05))
    rotation_sigma = float(data.get("rotation_sigma_deg", 3.0))
    if not all(np.isfinite(x) and x > 0 for x in (position_sigma, rotation_sigma)):
        raise ValueError("Prior standard deviations must be finite and positive")
    return Priors(
        cameras, poses, position_sigma, rotation_sigma, data.get("calibration")
    )


def align_to_pose_priors(model, priors):
    """Align the SfM gauge using both orientation and center priors, keeping units."""
    images = [
        im for im in model.images.values() if im.has_pose and im.name in priors.poses
    ]
    if len(images) < 3:
        raise ValueError(
            "At least three registered cameras with pose priors are required"
        )
    transform = _pose_similarity(images, priors)
    model.transform(transform)
    return float(transform.scale)


def _pose_similarity(images, priors):
    orientations = [
        priors.poses[im.name][:3, :3] @ im.cam_from_world().rotation.matrix()
        for im in images
    ]
    u, _, vt = np.linalg.svd(np.sum(orientations, axis=0))
    d = np.eye(3)
    d[2, 2] = np.linalg.det(u @ vt)
    rotation = u @ d @ vt
    a = np.stack([im.projection_center() for im in images])
    b = np.stack([priors.poses[im.name][:3, 3] for im in images])
    ac, bc = a - a.mean(0), b - b.mean(0)
    denom = np.sum(ac**2)
    if denom < 1e-12 or np.sum(bc**2) < 1e-12:
        raise ValueError(
            "Pose priors need a nonzero camera baseline to constrain scale"
        )
    scale = np.sum((ac @ rotation.T) * bc) / denom
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            "Pose priors and reconstruction have incompatible scale/orientations"
        )
    translation = b.mean(0) - scale * rotation @ a.mean(0)
    return pycolmap.Sim3d(scale, pycolmap.Rotation3d(rotation), translation)


def _prepare_prior_frame(model, priors):
    """Keep a supported metric gauge; recover an unaligned gauge from camera centers."""
    images = [
        im for im in model.images.values() if im.has_pose and im.name in priors.poses
    ]
    if len(images) < 3:
        raise ValueError("At least three registered pose priors are required")
    centers = np.stack([im.projection_center() for im in images])
    targets = np.stack([priors.poses[im.name][:3, 3] for im in images])
    # Allow for initial visual error without relaxing the BA prior weights.
    scene_radius = np.median(
        np.linalg.norm(targets - np.median(targets, axis=0), axis=1)
    )
    measurement_radius = np.sqrt(7.815) * priors.position_sigma
    radius = max(measurement_radius, 0.05 * scene_radius)
    required = max(3, len(images) // 2 + 1)
    distances = np.linalg.norm(centers - targets, axis=1)
    if np.max(np.linalg.norm(targets - targets.mean(0), axis=1)) < 1e-6:
        raise ValueError(
            "Pose priors need a nonzero camera baseline to constrain scale"
        )
    if np.count_nonzero(distances <= measurement_radius) >= required:
        return "preserved", 1.0
    # A straight path leaves rotation about the trajectory unobservable from centers.
    if np.linalg.matrix_rank(centers - centers.mean(0), tol=1e-6) < 2:
        transform = _pose_similarity(images, priors)
        action = "recovered_from_centers_and_orientations"
    else:
        action = "recovered_from_centers"
        options = pycolmap.RANSACOptions(max_error=radius, random_seed=0)
        transform = pycolmap.align_reconstruction_to_locations(
            model, [im.name for im in images], targets, 3, options
        )
    if transform is None:
        raise ValueError(
            "Cannot establish a metric frame consistent with the position priors"
        )
    aligned = transform * centers
    residual = np.linalg.norm(aligned - targets, axis=1)
    inliers = residual <= radius
    if np.count_nonzero(inliers) < required:
        raise ValueError(
            "Cannot establish a metric frame consistent with the position priors"
        )
    # Distinguish a coherent gauge shift from scatter in the initial visual poses.
    correction = np.linalg.norm(aligned[inliers] - centers[inliers], axis=1)
    if np.median(correction) <= np.median(residual[inliers]):
        return "preserved", 1.0
    model.transform(transform)
    return action, float(transform.scale)


def refine_with_pose_priors(
    model, priors, *, iterations=3, fix_intrinsics=False, pixel_sigma=1.0, num_threads=4
):
    """Alternate soft full-pose refinement and COLMAP structure/intrinsics BA.

    Pixel reprojection, center and SO(3) prior residuals share a soft-L1 loss.
    The point step holds every camera pose fixed: it cannot normalize away scale.
    This is block-coordinate optimization, not a joint COLMAP SE(3)-prior BA.
    """
    if iterations < 1:
        raise ValueError("prior refinement iterations must be positive")
    frame_action, scale = _prepare_prior_frame(model, priors)
    rotation_sigma = np.deg2rad(priors.rotation_sigma_deg)
    for _ in range(iterations):
        for image in model.images.values():
            if not image.has_pose or image.name not in priors.poses:
                continue
            frame = model.frames[image.frame_id]
            if len(frame.data_ids) != 1:
                raise ValueError(
                    "Pose-prior refinement currently supports one camera per frame"
                )
            points = [p for p in image.points2D if p.has_point3D()]
            if len(points) < 6:
                continue
            xyz = np.stack([model.points3D[p.point3D_id].xyz for p in points])
            xy = np.stack([p.xy for p in points])
            camera = model.cameras[image.camera_id]
            prior = priors.poses[image.name]
            base_r = image.cam_from_world().rotation.matrix().T
            base_c = image.projection_center()

            def residual(delta):
                r = Rotation.from_rotvec(delta[:3]).as_matrix() @ base_r
                center = base_c + delta[3:]
                cam_xyz = (xyz - center) @ r
                predicted = camera.img_from_cam(cam_xyz)
                reproj = np.nan_to_num(
                    (predicted - xy) / pixel_sigma, nan=1e4, posinf=1e4, neginf=-1e4
                ).ravel()
                orientation = (
                    Rotation.from_matrix(prior[:3, :3].T @ r).as_rotvec()
                    / rotation_sigma
                )
                position = (center - prior[:3, 3]) / priors.position_sigma
                return np.concatenate((reproj, orientation, position))

            fit = least_squares(
                residual,
                np.zeros(6),
                loss="soft_l1",
                max_nfev=30,
                ftol=1e-6,
                xtol=1e-6,
                gtol=1e-6,
            )
            if (
                not np.isfinite(fit.x).all()
                or fit.cost
                > 0.5 * np.sum(2 * (np.sqrt(1 + residual(np.zeros(6)) ** 2) - 1)) + 1e-6
            ):
                continue
            r = Rotation.from_rotvec(fit.x[:3]).as_matrix() @ base_r
            center = base_c + fit.x[3:]
            frame.set_cam_from_world(
                image.camera_id,
                pycolmap.Rigid3d(pycolmap.Rotation3d(r.T), -r.T @ center),
            )
        ba = pycolmap.BundleAdjustmentOptions()
        ba.refine_rig_from_world = False
        ba.refine_sensor_from_rig = False
        ba.refine_focal_length = not fix_intrinsics
        ba.refine_extra_params = not fix_intrinsics
        ba.loss_function_type = pycolmap.LossFunctionType.SOFT_L1
        ba.loss_function_scale = pixel_sigma
        ba.solver_options.num_threads = num_threads
        ba.solver_options.max_num_iterations = 30
        ba.print_summary = False
        pycolmap.bundle_adjustment(model, ba)
    # Pose changes can invalidate observations retained by the original mapper.
    bad = []
    for image in model.images.values():
        if not image.has_pose:
            continue
        observed = [(i, p) for i, p in enumerate(image.points2D) if p.has_point3D()]
        if not observed:
            continue
        xyz = np.stack([model.points3D[p.point3D_id].xyz for _, p in observed])
        camera_xyz = (
            xyz @ image.cam_from_world().rotation.matrix().T
            + image.cam_from_world().translation
        )
        projected = model.cameras[image.camera_id].img_from_cam(camera_xyz)
        xy = np.stack([p.xy for _, p in observed])
        invalid = (
            (camera_xyz[:, 2] <= 0)
            | ~np.isfinite(projected).all(axis=1)
            | (np.linalg.norm(projected - xy, axis=1) > 4 * pixel_sigma)
        )
        bad.extend((image.image_id, observed[j][0]) for j in np.flatnonzero(invalid))
    before_points = model.num_points3D()
    for image_id, index in bad:
        if model.images[image_id].points2D[index].has_point3D():
            model.delete_observation(image_id, index)
    if model.num_points3D() == 0:
        raise ValueError(
            "Pose-prior refinement left no valid points; check prior conventions and weights"
        )
    model.update_point_3d_errors()
    return {
        "prior_frame_action": frame_action,
        "alignment_scale": scale,
        "iterations": iterations,
        "observations_rejected_after_refinement": len(bad),
        "points_removed_after_refinement": before_points - model.num_points3D(),
        "position_sigma_m": priors.position_sigma,
        "rotation_sigma_deg": priors.rotation_sigma_deg,
    }
