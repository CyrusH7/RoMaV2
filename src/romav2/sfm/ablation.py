"""Prepare and evaluate the image-only 7-Scenes prior experiment."""

import argparse
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pycolmap
from PIL import Image
from scipy.spatial.transform import Rotation

from .calibration import OFFICE_CALIBRATION, load_calibration, validate_pinhole
from .io import RunRecord, write_json

ARMS = ["baseline", "intrinsics", "extrinsics", "both"]


SEVEN_SCENES_RIG = Path(__file__).with_name("configs") / "7scenes_rig.json"


def read_depth_from_rgb(path):
    rig = json.loads(Path(path).read_text())
    transform = np.asarray(rig.get("depth_from_rgb"), dtype=float)
    if (
        rig.get("units") != "meters"
        or transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8)
        or not np.allclose(
            transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6
        )
        or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-6)
    ):
        raise ValueError(
            "Rig calibration requires a rigid 4x4 depth_from_rgb in meters"
        )
    return transform, rig


def prepare(args):
    if args.count < 3 or args.window < 1:
        raise ValueError("Need at least three frames and a positive pairing window")
    if not all(
        np.isfinite(x)
        for x in (
            args.noise_m,
            args.noise_deg,
            args.position_sigma,
            args.rotation_sigma_deg,
        )
    ):
        raise ValueError("Prior noise/weights must be finite")
    if min(args.position_sigma, args.rotation_sigma_deg) <= 0:
        raise ValueError("Prior standard deviations must be positive")
    if min(args.noise_m, args.noise_deg) < 0:
        raise ValueError("Prior noise cannot be negative")
    depth_from_rgb, rig = read_depth_from_rgb(
        getattr(args, "rig_calibration", SEVEN_SCENES_RIG)
    )
    calibration = load_calibration(args.calibration)
    if args.intrinsics is not None:
        calibration = type(calibration)(
            calibration.width,
            calibration.height,
            tuple(validate_pinhole(args.intrinsics)),
            "command_line_override",
        )
    args.intrinsics = list(calibration.params)
    args.output.mkdir(parents=True, exist_ok=False)
    images = args.output / "images"
    images.mkdir()
    source = args.dataset / f"{args.sequence}.zip"
    extracted = args.dataset / args.sequence
    run = RunRecord(args.output)
    run.data["stage"] = "data_preparation"
    run.save()
    archive = None
    try:
        archive = ZipFile(source) if source.exists() else None
        if archive:
            frames = sorted(n for n in archive.namelist() if n.endswith(".color.png"))
            read = archive.read
        else:
            frames = sorted(p.name for p in extracted.glob("*.color.png"))

            def read(name):
                return (extracted / name).read_bytes()

        if len(frames) < args.count:
            raise ValueError(
                f"Found {len(frames)} color frames, requested {args.count}"
            )
        selected = [
            frames[i] for i in np.linspace(0, len(frames) - 1, args.count, dtype=int)
        ]
        rng = np.random.default_rng(args.seed)
        gt, prior = {}, {}
        for frame in selected:
            name = Path(frame).name
            image_bytes = read(frame)
            with Image.open(io.BytesIO(image_bytes)) as image:
                calibration.validate_size(image.size)
            (images / name).write_bytes(image_bytes)
            pose = np.loadtxt(
                io.BytesIO(read(frame.replace(".color.png", ".pose.txt")))
            )
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(f"Invalid reference pose for {frame}")
            # Original rotations can have small numerical non-orthogonality.
            pose[:3, :3] = Rotation.from_matrix(pose[:3, :3]).as_matrix()
            # KinectFusion tracks the depth camera. RoMaV2 observes RGB pixels.
            pose = pose @ depth_from_rgb
            noisy = pose.copy()
            noisy[:3, 3] += rng.normal(0, args.noise_m, 3)
            noisy[:3, :3] = (
                Rotation.from_rotvec(
                    rng.normal(0, np.deg2rad(args.noise_deg), 3)
                ).as_matrix()
                @ pose[:3, :3]
            )
            gt[name] = {"camera_to_world": pose.tolist()}
            prior[name] = {
                "camera_to_world": noisy.tolist(),
                "intrinsics": args.intrinsics,
            }
        names = sorted(prior)
        pairs = [
            (names[i], names[j])
            for i in range(len(names))
            for j in range(i + 1, min(len(names), i + args.window + 1))
        ]
        pairs += [
            (names[i], names[i + 2 * args.window])
            for i in range(len(names) - 2 * args.window)
        ]
        pairs = sorted(set(pairs))
        (args.output / "pairs.txt").write_text("".join(f"{a} {b}\n" for a, b in pairs))
        common = {
            "pose_convention": "camera_to_world_opencv",
            "units": "meters",
            "pose_sensor": "rgb",
            "source_pose_sensor": "depth",
            "rig_calibration": rig,
        }
        write_json(args.output / "reference.json", dict(common, images=gt))
        prior_doc = dict(
            common,
            images=prior,
            position_sigma=args.position_sigma,
            rotation_sigma_deg=args.rotation_sigma_deg,
            calibration=calibration.as_dict(),
        )
        write_json(args.output / "priors.json", prior_doc)
        manifest = dict(vars(args))
        manifest = {
            k: str(v) if isinstance(v, Path) else v for k, v in manifest.items()
        }
        manifest.update(
            {
                "images": names,
                "pairs": len(pairs),
                "calibration": calibration.as_dict(),
                "calibration_note": "User-provided RGB calibration; provenance recorded, not independently re-calibrated.",
                "prior_note": "Reference-derived synthetic noisy priors, not independent ARKit measurements.",
                "rig_calibration": rig,
                "depth_used": False,
            }
        )
        write_json(args.output / "experiment.json", manifest)
        print(
            f"Prepared {len(names)} frames and {len(pairs)} fixed pairs in {args.output}"
        )
    except Exception as error:
        run.finish(error)
        raise
    else:
        run.finish()
    finally:
        if archive:
            archive.close()


def alignment(a, b, allow_scale):
    ac, bc = a - a.mean(0), b - b.mean(0)
    if np.sum(ac**2) < 1e-12:
        raise ValueError("Camera trajectory is degenerate for evaluation alignment")
    u, s, vt = np.linalg.svd(bc.T @ ac)
    d = np.eye(3)
    d[-1, -1] = np.linalg.det(u @ vt)
    r = u @ d @ vt
    scale = (s * np.diag(d)).sum() / np.sum(ac**2) if allow_scale else 1.0
    return scale, r, b.mean(0) - scale * r @ a.mean(0)


def trajectory_metrics(poses, reference):
    names = sorted(set(poses) & set(reference))
    if len(names) < 3:
        return {"evaluated_images": len(names)}
    a = np.stack([poses[n][:3, 3] for n in names])
    b = np.stack([reference[n][:3, 3] for n in names])
    result = {
        "evaluated_images": len(names),
        "raw_ate_rmse_m": float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))),
    }
    result["alignment_rank"] = int(np.linalg.matrix_rank(a - a.mean(0), tol=1e-8))
    if np.sum((a - a.mean(0)) ** 2) < 1e-12:
        result["alignment_status"] = "degenerate_stationary_trajectory"
        return result
    for kind, allow_scale in [("se3", False), ("sim3", True)]:
        scale, r, t = alignment(a, b, allow_scale)
        error = np.linalg.norm(scale * a @ r.T + t - b, axis=1)
        angles = [
            Rotation.from_matrix(
                reference[n][:3, :3].T @ r @ poses[n][:3, :3]
            ).magnitude()
            for n in names
        ]
        result.update(
            {
                f"{kind}_ate_rmse_m": float(np.sqrt(np.mean(error**2))),
                f"{kind}_rotation_mean_deg": float(np.rad2deg(angles).mean())
                if result["alignment_rank"] >= 2
                else None,
            }
        )
        if allow_scale:
            result["sim3_scale_correction"] = float(scale)
    relative_rotation, relative_translation = [], []
    scale = result["sim3_scale_correction"]
    for n, nxt in zip(names, names[1:]):
        est = np.linalg.inv(poses[n]) @ poses[nxt]
        ref = np.linalg.inv(reference[n]) @ reference[nxt]
        relative_rotation.append(
            np.rad2deg(Rotation.from_matrix(ref[:3, :3].T @ est[:3, :3]).magnitude())
        )
        relative_translation.append(np.linalg.norm(scale * est[:3, 3] - ref[:3, 3]))
    result["adjacent_rpe_rotation_mean_deg"] = float(np.mean(relative_rotation))
    result["adjacent_rpe_translation_sim3_mean_m"] = float(
        np.mean(relative_translation)
    )
    return result


def model_metrics(model, reference, total_images):
    poses = {}
    coverages = []
    for im in model.images.values():
        if not im.has_pose:
            continue
        pose = np.eye(4)
        pose[:3, :3] = im.cam_from_world().rotation.matrix().T
        pose[:3, 3] = im.projection_center()
        poses[im.name] = pose
        cam = model.cameras[im.camera_id]
        xy = np.array([p.xy for p in im.points2D if p.has_point3D()]).reshape(-1, 2)
        cells = np.floor(xy / [cam.width, cam.height] * [16, 12]).astype(int)
        valid = (
            (cells[:, 0] >= 0)
            & (cells[:, 0] < 16)
            & (cells[:, 1] >= 0)
            & (cells[:, 1] < 12)
        )
        coverages.append(len(np.unique(cells[valid], axis=0)) / (16 * 12))
    points = list(model.points3D.values())
    errors = np.array([p.error for p in points])
    tracks = np.array([len({e.image_id for e in p.track.elements}) for p in points])
    result = trajectory_metrics(poses, reference)
    result.update(
        {
            "registered_images": model.num_reg_images(),
            "total_images": total_images,
            "points3D": model.num_points3D(),
            "mean_reprojection_px": model.compute_mean_reprojection_error(),
            "p95_point_reprojection_px": float(np.percentile(errors, 95))
            if len(errors)
            else None,
            "mean_distinct_view_track_length": float(tracks.mean())
            if len(tracks)
            else None,
            "fraction_tracks_3plus_views": float((tracks >= 3).mean())
            if len(tracks)
            else None,
            "mean_image_grid_coverage": float(np.mean(coverages)),
            "duplicate_image_tracks": sum(
                p.track.length() != len({e.image_id for e in p.track.elements})
                for p in points
            ),
        }
    )
    return result


def _model_poses(model):
    poses = {}
    for image in model.images.values():
        if not image.has_pose:
            continue
        pose = np.eye(4)
        pose[:3, :3] = image.cam_from_world().rotation.matrix().T
        pose[:3, 3] = image.projection_center()
        poses[image.name] = pose
    return poses


def common_pose_metrics(poses_by_arm, reference, results):
    common_names = (
        sorted(set(reference).intersection(*(set(p) for p in poses_by_arm.values())))
        if poses_by_arm
        else []
    )
    common_results = {}
    for arm, poses in poses_by_arm.items():
        metrics = trajectory_metrics({n: poses[n] for n in common_names}, reference)
        if results[arm].get("raw_ate_rmse_m") is None:
            metrics["raw_ate_rmse_m"] = None
        common_results[arm] = metrics
    return common_names, common_results


def evaluate(args):
    root = args.output
    doc = json.loads((root / "reference.json").read_text())
    reference = {n: np.array(e["camera_to_world"]) for n, e in doc["images"].items()}
    prior = json.loads((root / "priors.json").read_text())
    results = {
        "input_prior": trajectory_metrics(
            {n: np.array(e["camera_to_world"]) for n, e in prior["images"].items()},
            reference,
        )
    }
    poses_by_arm = {}
    arms = (
        [*ARMS, "fixed_poses"]
        if getattr(args, "with_fixed_poses", False) or (root / "fixed_poses").is_dir()
        else ARMS
    )
    if getattr(args, "with_prior_init", False) or (root / "prior_init").is_dir():
        arms = [*arms, "prior_init"]
    for arm in arms:
        summary, state = root / arm / "summary.json", root / arm / "run.json"
        if not summary.exists() or (
            state.exists() and json.loads(state.read_text())["status"] != "complete"
        ):
            results[arm] = {
                "status": "failed_or_incomplete",
                "log": str(root / f"{arm}.log"),
            }
            continue
        try:
            info = json.loads(summary.read_text())
            models = info.get("models", [])
            if not models:
                results[arm] = {"status": "no_model"}
                continue
            best = max(models, key=lambda m: (m["registered_images"], m["points3D"]))
            model = pycolmap.Reconstruction(
                str(root / arm / "sparse" / str(best["id"]))
            )
            results[arm] = model_metrics(model, reference, len(reference))
            results[arm].update(
                status="ok", num_models=len(models), selected_model=best["id"]
            )
            poses_by_arm[arm] = _model_poses(model)
            refinement = (
                root / arm / "sparse" / str(best["id"]) / "prior_refinement.json"
            )
            if (
                arm in ["baseline", "intrinsics"]
                or not refinement.exists()
                or json.loads(refinement.read_text()).get("status")
                not in {"refined_in_prior_frame", "fixed_in_prior_frame"}
            ):
                results[arm]["raw_ate_rmse_m"] = None
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            results[arm] = {"status": "evaluation_failed", "error": str(error)}
            poses_by_arm.pop(arm, None)
    common_names, common_results = common_pose_metrics(poses_by_arm, reference, results)
    write_json(root / "metrics.json", results)
    write_json(
        root / "metrics_common.json",
        {
            "arms": list(poses_by_arm),
            "image_names": common_names,
            "status": "ok" if len(common_names) >= 3 else "insufficient_common_images",
            "results": common_results,
        },
    )
    columns = [
        "arm",
        "status",
        "registered_images",
        "evaluated_images",
        "points3D",
        "mean_reprojection_px",
        "p95_point_reprojection_px",
        "mean_distinct_view_track_length",
        "fraction_tracks_3plus_views",
        "mean_image_grid_coverage",
        "duplicate_image_tracks",
        "sim3_ate_rmse_m",
        "se3_ate_rmse_m",
        "raw_ate_rmse_m",
        "sim3_rotation_mean_deg",
        "sim3_scale_correction",
        "adjacent_rpe_rotation_mean_deg",
        "adjacent_rpe_translation_sim3_mean_m",
    ]
    for filename, rows in [
        ("comparison.csv", results),
        ("comparison_common.csv", common_results),
    ]:
        with (root / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for arm, values in rows.items():
                writer.writerow(dict(values, arm=arm))
    for arm, values in results.items():
        print(arm, json.dumps(values))
    print(
        f"Comparison: {root / 'comparison.csv'}; common-frame comparison: {root / 'comparison_common.csv'}"
    )
    return all(results[arm].get("status") == "ok" for arm in arms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--dataset", type=Path, required=True)
    prep.add_argument("--sequence", default="seq-01")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--count", type=int, default=100)
    prep.add_argument("--window", type=int, default=5)
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument("--noise-m", type=float, default=0.02)
    prep.add_argument("--noise-deg", type=float, default=2.0)
    prep.add_argument("--position-sigma", type=float, default=0.05)
    prep.add_argument("--rotation-sigma-deg", type=float, default=3.0)
    prep.add_argument(
        "--rig-calibration",
        type=Path,
        default=SEVEN_SCENES_RIG,
        help="7-Scenes depth_from_rgb rigid transform in meters",
    )
    prep.add_argument("--calibration", type=Path, default=OFFICE_CALIBRATION)
    prep.add_argument(
        "--intrinsics",
        nargs=4,
        type=float,
        help="Explicit override of calibration fx fy cx cy",
    )
    ev = subs.add_parser("evaluate")
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--with-prior-init", action="store_true")
    ev.add_argument("--with-fixed-poses", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif not evaluate(args):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
