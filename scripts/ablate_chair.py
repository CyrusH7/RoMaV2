"""Compare visual and prior-guided reconstruction on a Stray Scanner sequence."""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from functools import lru_cache
from itertools import combinations, islice
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pycolmap
from PIL import Image
from scipy.spatial.transform import Rotation

from romav2.sfm.colmap import reconstruct
from romav2.sfm.io import write_json
from romav2.sfm.matching import RomaMatcher
from romav2.sfm.pairs import normalize_pairs, pairs_from_poses
from romav2.sfm.priors import load_priors
from romav2.sfm.strayscanner import scanner_priors
from romav2.sfm.tracks import track_component_sizes

ROOT = Path(__file__).resolve().parents[1]
MODES = ("visual", "K", "T", "KT", "KT_init", "KT_init_lm")
DEFAULT_ARMS = ["colmap_sift"] + [
    f"{pairing}_{mode}" for pairing in ("control", "pose") for mode in MODES[:-1]
]
ARMS = DEFAULT_ARMS + ["control_KT_init_lm", "pose_KT_init_lm"]


def native_features(images, output, names, threads, seed, max_features, guided=False):
    output.mkdir()
    db = str(output / "database.db")
    ext = pycolmap.FeatureExtractionOptions()
    ext.num_threads = threads
    ext.sift.max_num_features = max_features
    match = pycolmap.FeatureMatchingOptions()
    match.num_threads = threads
    match.guided_matching = guided
    # The installed PyCOLMAP wheel can be CPU-only even on CUDA hosts.
    device = pycolmap.Device.auto
    pycolmap.extract_features(
        db,
        str(images),
        names,
        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model="PINHOLE",
        extraction_options=ext,
        device=device,
    )
    geom = pycolmap.TwoViewGeometryOptions()
    geom.ransac.random_seed = seed
    pycolmap.match_exhaustive(
        db, matching_options=match, verification_options=geom, device=device
    )
    return db


class IndexedSiftMatches:
    def __init__(self, database):
        self.database = database

    def describe(self):
        return {
            "type": "COLMAP SIFT",
            "raw_matches_database": self.database,
            "keypoints": "native COLMAP indices",
            "score": "inverse descriptor distance",
        }

    def load_indexed_matches(self, names, pairs, min_score):
        matches, scores, counts = {}, {}, {}
        with pycolmap.Database.open(self.database) as db:
            ids = {im.name: im.image_id for im in db.read_all_images()}
            xy = {n: db.read_keypoints(ids[n])[:, :2].copy() for n in names}
            descriptors = {n: db.read_descriptors(ids[n]) for n in names}
            for a, b in pairs:
                m = db.read_matches(ids[a], ids[b])
                delta = (
                    descriptors[a][m[:, 0]].astype(np.float32) - descriptors[b][m[:, 1]]
                )
                score = 1 / (1 + np.linalg.norm(delta, axis=1))
                keep = score > min_score
                matches[a, b], scores[a, b] = m[keep], score[keep]
                counts[f"{a} {b}"] = len(m)
        store = SimpleNamespace(
            keypoints=xy.__getitem__, matches=matches, scores=scores
        )
        return store, counts


def native_mapping(db, images, output, threads, seed):
    mapper = pycolmap.IncrementalPipelineOptions()
    mapper.random_seed = seed
    mapper.num_threads = threads
    mapper.min_model_size = 2
    sparse = output / "sparse"
    sparse.mkdir()
    models = pycolmap.incremental_mapping(db, str(images), str(sparse), options=mapper)
    if not models:
        raise RuntimeError("Native COLMAP produced no model")
    best = max(
        models, key=lambda i: (models[i].num_reg_images(), models[i].num_points3D())
    )
    return best, len(models)


def pair_lists(names, priors, top_k, control):
    poses = normalize_pairs(pairs_from_poses(names, priors.poses, top_k), names)
    if control == "exhaustive":
        return list(combinations(names, 2)), poses
    # Keep the temporal and pose-selected graphs at the same edge count.
    temporal = (
        (names[i], names[i + gap])
        for gap in range(1, len(names))
        for i in range(len(names) - gap)
    )
    return normalize_pairs(islice(temporal, len(poses)), names), poses


def run_arm(arm, args, names, priors, graphs, sift_match=None):
    images = args.dataset / "images"
    output = args.output / arm
    if arm == "colmap_sift" or getattr(args, "frontend", None) == "colmap-sift":
        if getattr(args, "native_database", None) is None:
            args.native_database = native_features(
                images,
                args.output / "colmap_sift",
                names,
                args.threads,
                args.seed,
                args.sift_features,
                args.native_guided_matching,
            )
        if arm == "colmap_sift":
            return native_mapping(
                args.native_database, images, output, args.threads, args.seed
            )
        sift_match = IndexedSiftMatches(args.native_database)
    pairing, mode = arm.split("_", 1)
    initialize = mode in ("KT_init", "KT_init_lm")
    use_k = initialize or mode in ("K", "KT")
    use_t = initialize or mode in ("T", "KT")
    matcher = sift_match or RomaMatcher(
        args.setting, args.max_matches, args.seed, args.match_cache
    )
    try:
        result = reconstruct(
            images,
            output,
            names,
            graphs[pairing],
            matcher,
            priors=priors if use_k or use_t else None,
            use_intrinsics_prior=use_k,
            fix_intrinsics=use_k,
            use_pose_prior=use_t,
            init_from_poses=initialize,
            refine_initial_pose=mode == "KT_init_lm",
            camera_model="PINHOLE",
            single_camera=not use_k,
            cell_size=args.cell_size,
            seed=args.seed,
            num_threads=args.threads,
        )
    finally:
        if sift_match is None:
            matcher.release_model()
    return result["best_model_id"], len(result["models"])


def save_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def visual_alignment(images, reference, radius, seed):
    names = [n for n, im in images.items() if im.num_points3D >= 6]
    if len(names) < 3:
        return None
    centers = np.array([images[n].projection_center() for n in names])
    targets = np.array([reference[n][:3, 3] for n in names])
    if any(
        np.linalg.matrix_rank(x - x.mean(0), tol=1e-7) < 2 for x in (centers, targets)
    ):
        return None
    transform = pycolmap.estimate_sim3d_robust(
        centers, targets, pycolmap.RANSACOptions(max_error=radius, random_seed=seed)
    )
    if isinstance(transform, dict):
        transform = transform["tgt_from_src"]
    if transform is None or not np.isfinite(transform.scale) or transform.scale <= 0:
        return None
    inliers = np.linalg.norm(transform * centers - targets, axis=1) <= radius
    if inliers.sum() < 3:
        return None
    return transform


def depth_errors(image, model, path, scale):
    if scale is None or not path.is_file():
        return np.empty(0)
    observations = [p for p in image.points2D if p.has_point3D()]
    if not observations:
        return np.empty(0)
    with Image.open(path) as png:
        depth = np.asarray(png)
    if depth.ndim != 2 or depth.dtype.kind not in "iu" or depth.dtype.itemsize < 2:
        raise ValueError(f"Expected a millimeter depth PNG at {path}")
    camera = model.cameras[image.camera_id]
    xy = np.array([p.xy for p in observations])
    uv = np.floor(
        xy * [depth.shape[1] / camera.width, depth.shape[0] / camera.height]
    ).astype(int)
    xyz = np.array([model.points3D[p.point3D_id].xyz for p in observations])
    z = (image.cam_from_world() * xyz)[:, 2] * scale
    inside = ((uv >= 0) & (uv < [depth.shape[1], depth.shape[0]])).all(axis=1) & (z > 0)
    measured = depth[uv[inside, 1], uv[inside, 0]].astype(float) / 1000
    valid = measured > 0
    return abs(z[inside][valid] - measured[valid])


def evaluate_arm(arm, args, names, reference, best, num_models):
    sparse = args.output / arm / "sparse"
    model = pycolmap.Reconstruction(str(sparse / str(best)))
    images = {im.name: im for im in model.images.values() if im.has_pose}
    memberships = {name: [best] if name in images else [] for name in names}
    if num_models > 1:
        for path in sorted(sparse.iterdir()):
            if not path.name.isdecimal() or int(path.name) == best or not path.is_dir():
                continue
            other = pycolmap.Reconstruction(str(path))
            for image_id in other.reg_image_ids():
                memberships[other.images[image_id].name].append(int(path.name))
    refine_path = sparse / str(best) / "prior_refinement.json"
    metric = (
        refine_path.exists()
        and json.loads(refine_path.read_text())["status"] == "refined_in_prior_frame"
    )
    transform = (
        pycolmap.Sim3d()
        if metric
        else visual_alignment(images, reference, args.align_inlier_m, args.seed)
    )
    rows, all_depth = [], []
    for name in names:
        row = {
            "arm": arm,
            "image": name,
            "registered": name in images,
            "model_ids": ",".join(map(str, sorted(memberships[name]))),
            "observations": 0,
        }
        rows.append(row)
        if name not in images:
            continue
        im = images[name]
        row["observations"] = im.num_points3D
        if transform is None:
            continue
        ref = reference[name]
        rotation = transform.rotation.matrix() @ im.cam_from_world().rotation.matrix().T
        position = np.linalg.norm(transform * im.projection_center() - ref[:3, 3])
        angle = np.degrees(Rotation.from_matrix(ref[:3, :3].T @ rotation).magnitude())
        row.update(
            center_m=position,
            rotation_deg=angle,
            flagged=bool(position > args.position_m or angle > args.rotation_deg),
        )
        errors = depth_errors(
            im,
            model,
            args.dataset / "depth" / Path(name).with_suffix(".png"),
            transform.scale,
        )
        if len(errors):
            row.update(depth_samples=len(errors), depth_median_abs_m=np.median(errors))
            all_depth.append(errors)
    valid = [r for r in rows if "flagged" in r and r["observations"] >= 6]
    evaluated = {r["image"] for r in valid}
    steps = []
    for a, b in zip(names, names[1:]):
        if a in evaluated and b in evaluated:
            relative = images[b].cam_from_world() * images[a].cam_from_world().inverse()
            ref = np.linalg.inv(reference[b]) @ reference[a]
            angle = Rotation.from_matrix(
                ref[:3, :3].T @ relative.rotation.matrix()
            ).magnitude()
            steps.append(
                (
                    np.linalg.norm(transform.scale * relative.translation - ref[:3, 3]),
                    np.degrees(angle),
                )
            )
    depth = np.concatenate(all_depth) if all_depth else []
    components = track_component_sizes(model)
    summary = dict(
        arm=arm,
        status="ok" if transform is not None else "pose_alignment_unavailable",
        models=num_models,
        model_id=best,
        total=len(names),
        registered=len(images),
        registered_any_model=sum(bool(ids) for ids in memberships.values()),
        largest_visual_component=max(components, default=0),
        visual_track_component_sizes=",".join(map(str, components)),
        supported=sum(r["observations"] >= 6 for r in rows),
        gauge="prior_meters" if metric else "robust_sim3_to_arkit",
        scale=transform.scale if transform else None,
        evaluated=len(valid),
        adjacent_pairs=len(steps),
        adjacent_translation_p95_m=np.percentile(np.asarray(steps)[:, 0], 95)
        if steps
        else None,
        adjacent_rotation_p95_deg=np.percentile(np.asarray(steps)[:, 1], 95)
        if steps
        else None,
        flagged=sum(r["flagged"] for r in valid) if valid else None,
        points=model.num_points3D(),
        reprojection_px=model.compute_mean_reprojection_error(),
        depth_samples=len(depth),
        depth_median_abs_m=float(np.median(depth)) if len(depth) else None,
    )
    if valid:
        summary.update(
            center_rmse_m=np.sqrt(np.mean([r["center_m"] ** 2 for r in valid])),
            rotation_p95_deg=np.percentile([r["rotation_deg"] for r in valid], 95),
        )
    return summary, rows


def comparisons(arms, by_arm):
    contrasts = [(f"control_{m}", f"pose_{m}") for m in MODES]
    for p in ("control", "pose"):
        contrasts.extend(
            (f"{p}_{a}", f"{p}_{b}")
            for a, b in (
                ("visual", "K"),
                ("visual", "T"),
                ("K", "KT"),
                ("T", "KT"),
                ("KT", "KT_init"),
                ("visual", "KT_init"),
                ("KT_init", "KT_init_lm"),
            )
        )
    contrasts.extend(("colmap_sift", b) for b in ("control_visual", "pose_KT_init"))
    results = []
    for a, b in contrasts:
        if a not in arms or b not in arms:
            continue
        if a not in by_arm or b not in by_arm:
            results.append(dict(before=a, after=b, status="missing_result"))
            continue
        left, right = by_arm[a], by_arm[b]

        def supported(rows):
            return {
                n for n, r in rows.items() if r["observations"] >= 6 and "flagged" in r
            }

        na, nb = supported(left), supported(right)
        common = sorted(na & nb)
        rec = dict(
            before=a,
            after=b,
            status="ok" if common else "no_common_evaluable_frames",
            common=len(common),
            before_only=len(na - nb),
            after_only=len(nb - na),
            flagged_lost=sum(left[n]["flagged"] for n in na - nb),
        )
        if common:
            rec.update(
                before_flagged=sum(left[n]["flagged"] for n in common),
                after_flagged=sum(right[n]["flagged"] for n in common),
                corrected=sum(
                    left[n]["flagged"] and not right[n]["flagged"] for n in common
                ),
                newly_flagged=sum(
                    right[n]["flagged"] and not left[n]["flagged"] for n in common
                ),
            )
        depth_frames = [
            n
            for n in common
            if "depth_median_abs_m" in left[n] and "depth_median_abs_m" in right[n]
        ]
        rec["common_depth_frames"] = len(depth_frames)
        if depth_frames:
            rec.update(
                before_frame_median_depth_abs_m=np.median(
                    [left[n]["depth_median_abs_m"] for n in depth_frames]
                ),
                after_frame_median_depth_abs_m=np.median(
                    [right[n]["depth_median_abs_m"] for n in depth_frames]
                ),
            )
        results.append(rec)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=ROOT / "data/chair")
    p.add_argument("--output", type=Path)
    p.add_argument("--match-cache", type=Path, default=ROOT / "runs/chair_match_cache")
    p.add_argument(
        "--frontend",
        choices=("roma", "sift", "colmap-sift"),
        default="roma",
        help="Matcher for the controlled arms; native baseline always uses COLMAP SIFT",
    )
    p.add_argument("--arms", nargs="+", choices=ARMS, default=DEFAULT_ARMS)
    p.add_argument("--control", choices=("temporal", "exhaustive"), default="temporal")
    p.add_argument(
        "--frames", type=int, help="First N RGB frames in numeric order (default: all)"
    )
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--position-m", type=float, default=0.20)
    p.add_argument(
        "--align-inlier-m",
        type=float,
        default=0.20,
        help="RANSAC inlier radius for gauge-free trajectory alignment",
    )
    p.add_argument(
        "--position-sigma-m", type=float, default=0.05, help="Soft position prior sigma"
    )
    p.add_argument(
        "--rotation-sigma-deg",
        type=float,
        default=3.0,
        help="Soft orientation prior sigma",
    )
    p.add_argument("--rotation-deg", type=float, default=15)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--native-guided-matching",
        action="store_true",
        help="Enable COLMAP geometry-guided SIFT matching for the native baseline",
    )
    p.add_argument("--sift-features", type=int, default=4096)
    p.add_argument("--max-matches", type=int, default=1500)
    p.add_argument("--cell-size", type=float, default=2)
    p.add_argument(
        "--setting", choices=("turbo", "fast", "base", "precise"), default="fast"
    )
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    if any(
        not np.isfinite(v) or v <= 0
        for v in (
            args.position_m,
            args.rotation_deg,
            args.align_inlier_m,
            args.position_sigma_m,
            args.rotation_sigma_deg,
            args.threads,
            args.top_k,
            args.sift_features,
            args.max_matches,
            args.cell_size,
        )
    ):
        p.error("Thresholds, counts and cell size must be positive")
    if len(set(args.arms)) != len(args.arms):
        p.error("Each arm must be specified once")
    args.dataset = args.dataset.resolve()
    args.match_cache = args.match_cache.resolve()
    args.output = (
        args.output
        or ROOT / "runs" / datetime.now().strftime("chair_ablation_%Y%m%d_%H%M%S_%f")
    ).resolve()
    if args.output.exists():
        p.error("Choose a new output directory")
    if (
        not args.prepare_only
        and args.frontend == "roma"
        and any(a != "colmap_sift" for a in args.arms)
    ):
        import torch

        if not torch.cuda.is_available():
            p.error(
                "RoMaV2 arms require an active GPU. Use --prepare-only or --arms colmap_sift on CPU."
            )
    doc = scanner_priors(args.dataset)
    names = sorted(doc["images"], key=lambda n: int(Path(n).stem))
    if args.frames is not None and not 3 <= args.frames <= len(names):
        p.error(f"--frames must be between 3 and {len(names)}")
    names = names[: args.frames]
    doc["images"] = {n: doc["images"][n] for n in names}
    doc.update(
        position_sigma=args.position_sigma_m, rotation_sigma_deg=args.rotation_sigma_deg
    )
    args.output.mkdir(parents=True)
    write_json(args.output / "priors.json", doc)
    priors = load_priors(args.output / "priors.json", names)
    control, pose = pair_lists(names, priors, args.top_k, args.control)
    graphs = dict(control=control, pose=pose)
    for key, pairs in graphs.items():
        (args.output / f"{key}_pairs.txt").write_text(
            "".join(f"{a} {b}\n" for a, b in pairs)
        )
    write_json(
        args.output / "experiment.json",
        dict(
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            images=names,
            control_pairs=len(control),
            pose_pairs=len(pose),
            shared_pairs=len(set(pose) & set(control)),
            native_pairs=len(names) * (len(names) - 1) // 2,
        ),
    )
    print(
        f"Results: {args.output}\n{len(names)} frames, {len(control)} control pairs, {len(pose)} pose-selected pairs",
        flush=True,
    )
    if set(control) == set(pose):
        print("Both pair graphs are identical for this subset and K.", flush=True)
    if args.prepare_only:
        return
    sift_match = None
    if args.frontend == "sift" and any(a != "colmap_sift" for a in args.arms):
        from check_colmap_sift import sift_matcher
        import cv2

        cv2.setNumThreads(1)
        sift_match = lru_cache(maxsize=None)(
            sift_matcher(args.dataset / "images", names, max_features=700, downsample=2)
        )
    summaries, frames, by_arm = [], [], {}
    failed = False
    for arm in args.arms:
        print(f"Running {arm}", flush=True)
        try:
            best, count = run_arm(arm, args, names, priors, graphs, sift_match)
            summary, rows = evaluate_arm(arm, args, names, priors.poses, best, count)
            frames.extend(rows)
            by_arm[arm] = {r["image"]: r for r in rows}
        except (ValueError, OSError, RuntimeError) as error:
            logging.exception("Arm %s failed", arm)
            failed = True
            summary = dict(arm=arm, status="failed", error=str(error))
        summary["frontend"] = "native_sift" if arm == "colmap_sift" else args.frontend
        summaries.append(summary)
        save_csv(args.output / "comparison.csv", summaries)
        if frames:
            save_csv(args.output / "frames.csv", frames)
        contrasts = comparisons(args.arms, by_arm)
        if contrasts:
            save_csv(args.output / "contrasts.csv", contrasts)
        print(summary, flush=True)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
