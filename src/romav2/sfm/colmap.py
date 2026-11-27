"""A staged pipeline from external correspondences to COLMAP incremental SfM."""

import hashlib
import logging
import platform
from contextlib import closing
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from shutil import copyfile
from sqlite3 import connect
from tempfile import TemporaryDirectory

import numpy as np
import pycolmap
from PIL import Image

from .calibration import calibration_from_dict, validate_pinhole
from .database import create_database, verify_database
from .io import RunRecord, write_json
from .matching import FeatureStore
from .options import PipelineOptions
from .pairs import normalize_pairs, pairs_from_poses
from .tracks import track_component_sizes

LOG = logging.getLogger(__name__)


def _preflight(images, output, names, pairs, options, camera_params, priors):
    if not images.is_dir():
        raise ValueError(f"Image directory does not exist: {images}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output directory must be empty: {output}")
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("At least two unique image names are required")
    for name in names:
        path = Path(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != name
            or any(c.isspace() for c in name)
            or not (images / name).resolve().is_relative_to(images)
        ):
            raise ValueError(
                f"Image name must be a relative POSIX path without whitespace: {name}"
            )
    pairs = normalize_pairs(pairs, names)
    if options.use_intrinsics_prior and (
        priors is None or set(names) - priors.cameras.keys()
    ):
        raise ValueError("Every selected image must have an intrinsics prior")
    if options.use_pose_prior and (priors is None or set(names) - priors.poses.keys()):
        raise ValueError("Every selected image must have a pose prior")
    if options.use_pose_prior and not options.fix_poses and len(names) < 3:
        raise ValueError("Soft pose refinement requires at least three images")
    if (
        options.fix_intrinsics
        and camera_params is None
        and not options.use_intrinsics_prior
    ):
        raise ValueError("--fix-intrinsics requires known calibration")
    if options.use_intrinsics_prior and camera_params is not None:
        raise ValueError("Use either per-image prior intrinsics or --camera-params")
    sizes = {}
    for name in names:
        with Image.open(images / name) as image:
            sizes[name] = image.size
    if camera_params is not None:
        camera_params = validate_pinhole(camera_params)
    if (camera_params is not None or options.single_camera) and len(
        set(sizes.values())
    ) != 1:
        raise ValueError("A shared camera requires images of identical dimensions")
    if options.use_intrinsics_prior:
        for name in names:
            validate_pinhole(priors.cameras[name])
        if priors.calibration is not None:
            declared = calibration_from_dict(priors.calibration)
            for size in sizes.values():
                declared.validate_size(size)
    return pairs, sizes, camera_params


def _mapping_options(options):
    mapper = pycolmap.IncrementalPipelineOptions()
    mapper.min_num_matches = options.min_matches
    mapper.min_model_size = 2
    mapper.random_seed = options.seed
    mapper.num_threads = options.num_threads
    mapper.use_prior_position = options.use_pose_prior
    mapper.use_robust_loss_on_prior_position = options.use_pose_prior
    mapper.ba_refine_focal_length = not options.fix_intrinsics
    mapper.ba_refine_extra_params = not options.fix_intrinsics
    mapper.mapper.abs_pose_refine_focal_length = not options.fix_intrinsics
    mapper.mapper.abs_pose_refine_extra_params = not options.fix_intrinsics
    return mapper


class _InsufficientPriorGeometry(RuntimeError):
    pass


def _triangulate_known_poses(db_path, images, sparse, priors, options):
    """Use the database IDs throughout so verified match indices stay valid."""
    model = pycolmap.Reconstruction()
    with pycolmap.Database.open(str(db_path)) as db:
        for camera in db.read_all_cameras():
            model.add_camera(camera)
        for rig in db.read_all_rigs():
            model.add_rig(rig)
        frame_ids = {}
        for frame in db.read_all_frames():
            model.add_frame(frame)
            for image_id in frame.image_ids:
                frame_ids[image_id.id] = frame.frame_id
        db_images = db.read_all_images()
        frame_names = [image.name for image in db_images]
        for image in db_images:
            image.frame_id = frame_ids[image.image_id]
            model.add_image(image)
            model.frames[image.frame_id].set_cam_from_world(
                image.camera_id,
                priors.cam_from_world(image.name),
            )
            model.register_frame(image.frame_id)
    mapper = _mapping_options(options)
    mapper.use_prior_position = False
    mapper.triangulation.ignore_two_view_tracks = False
    model = pycolmap.triangulate_points(
        model,
        str(db_path),
        str(images),
        str(sparse / "0"),
        options=mapper,
        refine_intrinsics=False,
    )
    if model.num_points3D() == 0:
        raise _InsufficientPriorGeometry(
            "Fixed camera poses produced no triangulated points"
        )
    if options.init_from_poses:
        before = {
            im.name: im.num_points3D for im in model.images.values() if im.has_pose
        }
        # Removing a weak view can shorten tracks in the remaining views.
        while True:
            weak = [
                model.images[i].frame_id
                for i in model.reg_image_ids()
                if model.images[i].num_points3D < 6
            ]
            if not weak:
                break
            for frame_id in weak:
                model.deregister_frame(frame_id)
        retained = {model.images[i].name for i in model.reg_image_ids()}
        write_json(
            sparse.parent / "prior_triangulation.json",
            {
                "min_observations": 6,
                "images": {
                    name: {
                        "initial_points": before.get(name, 0),
                        "retained": name in retained,
                    }
                    for name in frame_names
                },
            },
        )
        if model.num_reg_images() < 3:
            raise _InsufficientPriorGeometry(
                "Prior initialization needs three views with six triangulated observations each"
            )
    model.update_point_3d_errors()
    return {0: model}


def _validate_refined_map(
    db_path, images, sparse, priors, candidate, options, verify, report
):
    with TemporaryDirectory(dir=sparse.parent) as directory:
        baseline = Path(directory)
        baseline_db = baseline / "database.db"
        baseline_sparse = baseline / "sparse"
        baseline_sparse.mkdir()
        with (
            closing(connect(db_path)) as source,
            closing(connect(baseline_db)) as destination,
        ):
            source.backup(destination)
        baseline_stats = verify(priors, baseline_db)

        baseline_models = None
        try:
            baseline_models = _triangulate_known_poses(
                baseline_db, images, baseline_sparse, priors, options
            )
        except _InsufficientPriorGeometry:
            pass

        refined_stats = verify(candidate, db_path)
        refined_models = None
        try:
            refined_models = _triangulate_known_poses(
                db_path, images, sparse, candidate, options
            )
        except _InsufficientPriorGeometry:
            if baseline_models is None:
                raise

        baseline_size = (
            max(track_component_sizes(baseline_models[0]), default=0)
            if baseline_models
            else 0
        )
        refined_size = (
            max(track_component_sizes(refined_models[0]), default=0)
            if refined_models
            else 0
        )
        report["initial_map_largest_component"] = {
            "input_poses": baseline_size,
            "refined_poses": refined_size,
        }
        if refined_size >= baseline_size:
            return refined_models, refined_stats

        with (
            closing(connect(baseline_db)) as source,
            closing(connect(db_path)) as destination,
        ):
            source.backup(destination)
        copyfile(
            baseline / "prior_triangulation.json",
            sparse.parent / "prior_triangulation.json",
        )
        model_path = sparse / "0"
        model_path.mkdir(exist_ok=True)
        baseline_models[0].write(str(model_path))
        report.update(status="kept_input_poses", reason="refined_map_lost_connectivity")
        return baseline_models, baseline_stats


def _snapshot_priors(priors, names, options):
    images = {}
    for name in names:
        entry = {}
        if options.use_intrinsics_prior:
            entry["intrinsics"] = np.asarray(priors.cameras[name]).tolist()
        if options.use_pose_prior:
            entry["camera_to_world"] = np.asarray(priors.poses[name]).tolist()
        images[name] = entry
    return {
        "pose_convention": "camera_to_world_opencv",
        "units": "meters",
        "position_sigma": priors.position_sigma,
        "rotation_sigma_deg": priors.rotation_sigma_deg,
        "calibration": priors.calibration,
        **(
            {"pose_sensor": "rgb", "rig_calibration": priors.rig_calibration}
            if priors.rig_calibration and options.use_pose_prior
            else {}
        ),
        "images": images,
    }


def reconstruct(
    images,
    output,
    names,
    pairs,
    matcher,
    *,
    cell_size=1.0,
    camera_params=None,
    min_matches=15,
    fix_intrinsics=False,
    seed=0,
    priors=None,
    use_intrinsics_prior=False,
    use_pose_prior=False,
    prior_iterations=3,
    joint_prior_ba=False,
    camera_model=None,
    num_threads=4,
    min_match_score=0.0,
    geometric_max_error=4.0,
    geometric_max_trials=20000,
    geometric_min_inlier_ratio=0.1,
    calibration=None,
    fix_poses=False,
    init_from_poses=False,
    pose_select_k=None,
    single_camera=False,
    refine_initial_pose=False,
):
    use_pose_prior = use_pose_prior or fix_poses or init_from_poses
    options = PipelineOptions(
        fix_poses=fix_poses,
        init_from_poses=init_from_poses,
        single_camera=single_camera,
        refine_initial_pose=refine_initial_pose,
        cell_size=cell_size,
        min_matches=min_matches,
        fix_intrinsics=fix_intrinsics,
        seed=seed,
        use_intrinsics_prior=use_intrinsics_prior,
        use_pose_prior=use_pose_prior,
        prior_iterations=prior_iterations,
        joint_prior_ba=joint_prior_ba,
        camera_model=camera_model,
        num_threads=num_threads,
        min_match_score=min_match_score,
        geometric_max_error=geometric_max_error,
        geometric_max_trials=geometric_max_trials,
        geometric_min_inlier_ratio=geometric_min_inlier_ratio,
    )
    images, output = Path(images).resolve(), Path(output).resolve()
    names = sorted(names)
    if pose_select_k is not None:
        if pairs is not None or priors is None:
            raise ValueError(
                "Pose selection requires priors and cannot use an explicit pair list"
            )
        pairs = pairs_from_poses(names, priors.poses, pose_select_k)
        pose_select_k = int(pose_select_k)
    if calibration is not None:
        if camera_params is not None or use_intrinsics_prior:
            raise ValueError(
                "Calibration file cannot be combined with other intrinsics inputs"
            )
        for name in names:
            with Image.open(images / name) as image:
                calibration.validate_size(image.size)
        camera_params = calibration.params
    pairs, sizes, camera_params = _preflight(
        images, output, names, pairs, options, camera_params, priors
    )
    LOG.info("Selected %d images and %d unique pairs", len(names), len(pairs))
    output.mkdir(parents=True, exist_ok=True)
    run = RunRecord(output)
    try:
        write_json(
            output / "config.json",
            {
                **asdict(options),
                "images": str(images),
                "pose_select_k": pose_select_k,
                "camera_params": None
                if camera_params is None
                else camera_params.tolist(),
                "versions": {
                    "pycolmap": pycolmap.__version__,
                    "numpy": np.__version__,
                    "python": platform.python_version(),
                    "scipy": version("scipy"),
                },
                "backend_code_digest": hashlib.sha256(
                    b"".join(
                        p.name.encode() + p.read_bytes()
                        for p in sorted(Path(__file__).parent.glob("*.py"))
                    )
                ).hexdigest(),
                "calibration": calibration.as_dict() if calibration else None,
                "matcher": matcher.describe()
                if hasattr(matcher, "describe")
                else {"type": type(matcher).__name__},
            },
        )
        write_json(
            output / "images.json",
            [{"name": n, "width": sizes[n][0], "height": sizes[n][1]} for n in names],
        )
        if use_intrinsics_prior or use_pose_prior:
            write_json(
                output / "priors_used.json", _snapshot_priors(priors, names, options)
            )
        pairs_path = output / "pairs.txt"
        pairs_path.write_text("".join(f"{a} {b}\n" for a, b in pairs), encoding="utf-8")
        with run.stage("matching"):
            if hasattr(matcher, "load_indexed_matches"):
                store, sampled_counts = matcher.load_indexed_matches(
                    names, pairs, min_match_score
                )
            else:
                store = FeatureStore(cell_size, min_match_score=min_match_score)
                sampled_counts = {}
                for index, (a, b) in enumerate(pairs, 1):
                    LOG.info("Matching %d/%d: %s <-> %s", index, len(pairs), a, b)
                    xy_a, xy_b, scores = matcher(images / a, images / b)
                    sampled_counts[f"{a} {b}"] = len(scores)
                    store.add(a, b, xy_a, xy_b, scores, sizes)
            raw_counts = {f"{a} {b}": len(m) for (a, b), m in store.matches.items()}
            write_json(
                output / "matching.json",
                {"pairs": raw_counts, "statistics": getattr(matcher, "statistics", {})},
            )
            if hasattr(matcher, "release_model"):
                matcher.release_model()
        db_path = output / "database.db"
        with run.stage("database_import"):
            ids, cameras = create_database(
                db_path, images, names, sizes, store, options, camera_params, priors
            )
            write_json(output / "cameras_initial.json", cameras)
        initial_priors = priors
        if options.refine_initial_pose:
            from .initialization import refine_initial_pair
            from .pose_graph import refine_pose_graph

            initialize = (
                refine_pose_graph
                if options.refine_initial_pose == "graph"
                else refine_initial_pair
            )

            with run.stage("initial_pose_refinement"):
                initial_priors, refinement = initialize(
                    db_path, pairs, ids, priors, options
                )
                write_json(output / "initial_pose_refinement.json", refinement)

        def verify(geometry_priors, database_path=db_path, match_store=store):
            return verify_database(
                database_path,
                pairs_path,
                names,
                pairs,
                ids,
                match_store,
                options,
                geometry_priors,
            )

        sparse = output / "sparse"
        sparse.mkdir()
        models = None
        if initial_priors is priors:
            with run.stage("geometric_verification"):
                verified, pair_counts, conflicts = verify(priors)
        else:
            with run.stage("validate_initial_map"):
                models, (verified, pair_counts, conflicts) = _validate_refined_map(
                    db_path,
                    images,
                    sparse,
                    priors,
                    initial_priors,
                    options,
                    verify,
                    refinement,
                )
            if refinement["status"] != "refined":
                initial_priors = priors
            write_json(output / "initial_pose_refinement.json", refinement)

        with run.stage("pair_summary"):
            write_json(
                output / "pair_statistics.json",
                {
                    key: {
                        "sampled_matches": sampled_counts[key],
                        "raw_matches": raw_counts[key],
                        "geometric_inliers": verified[key],
                        "track_consistent_inliers": pair_counts[key],
                    }
                    for key in raw_counts
                },
            )
        del verify, store
        if models is None and options.triangulate_from_priors:
            with run.stage(
                "prior_pose_triangulation"
                if init_from_poses
                else "fixed_pose_triangulation"
            ):
                models = _triangulate_known_poses(
                    db_path, images, sparse, initial_priors, options
                )
        elif models is None:
            with run.stage("incremental_mapping"):
                models = pycolmap.incremental_mapping(
                    str(db_path),
                    str(images),
                    str(sparse),
                    options=_mapping_options(options),
                )
        info = {
            "images": len(names),
            "pairs": len(pairs),
            "verified_pairs": sum(v >= min_matches for v in pair_counts.values()),
            "verified_inliers": pair_counts,
            "track_conflicts_rejected": conflicts,
            "models": [],
        }
        with run.stage("model_finalization"):
            for model_id, model in sorted(models.items()):
                if use_pose_prior:
                    from .priors import refine_with_pose_priors

                    if fix_poses:
                        refinement = {
                            "status": "fixed_in_prior_frame",
                            "alignment_scale": 1.0,
                        }
                    elif model.num_reg_images() >= 3:
                        refinement = refine_with_pose_priors(
                            model,
                            priors,
                            iterations=prior_iterations,
                            joint_prior_ba=joint_prior_ba,
                            fix_intrinsics=fix_intrinsics,
                            num_threads=num_threads,
                        )
                        refinement["status"] = "refined_in_prior_frame"
                        refinement["initialization"] = (
                            "input_poses" if init_from_poses else "incremental_mapper"
                        )
                        if initial_priors is not priors:
                            refinement["initialization"] = {
                                "joint": "sampson_lm_joint",
                                "landmarks": "external_landmarks",
                                "graph": "raw_match_pose_graph",
                            }.get(
                                options.refine_initial_pose, "sampson_lm_fixed_centers"
                            )
                    else:
                        refinement = {
                            "status": "skipped_fewer_than_three_registered_images"
                        }
                    if init_from_poses:
                        from .registration import recover_prior_views

                        recovery = recover_prior_views(
                            model, db_path, pairs, priors, options
                        )
                        write_json(
                            sparse / str(model_id) / "prior_registration.json", recovery
                        )
                    model.write(str(sparse / str(model_id)))
                    write_json(
                        sparse / str(model_id) / "prior_refinement.json", refinement
                    )
                info["models"].append(
                    {
                        "id": int(model_id),
                        "registered_images": model.num_reg_images(),
                        "points3D": model.num_points3D(),
                        "mean_reprojection_error": model.compute_mean_reprojection_error(),
                    }
                )
            info["best_model_id"] = (
                max(
                    info["models"],
                    key=lambda m: (m["registered_images"], m["points3D"]),
                )["id"]
                if models
                else None
            )
            write_json(output / "summary.json", info)
            if not models:
                raise RuntimeError(
                    f"COLMAP did not reconstruct a model; see {output}/summary.json"
                )
    except Exception as error:
        run.finish(error)
        raise
    run.finish()
    return info
