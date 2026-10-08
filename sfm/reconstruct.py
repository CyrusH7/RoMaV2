"""RoMaV2 dense matching -> hloc keypoint aggregation -> COLMAP SfM (cameras/images/points3D .bin).

Pipeline (mirrors hloc's `match_dense` workflow, with RoMaV2 as the dense matcher):

    images -> pairs.txt (ARKit-pose-guided: rotation pre-filter + nearest camera centers, top-20 per image;
              NetVLAD retrieval is skipped whenever poses are available)
           -> RoMaV2 dense matches per pair (raw_matches/*.npz)
           -> hloc quantize/aggregate into per-image keypoints + matches0 (feats.h5 / matches.h5)
           -> pycolmap: database, geometric verification, mapping / triangulation
           -> <output_dir>/{cameras,images,points3D,frames,rigs}.bin

Directory layout (everything not given explicitly is derived from --data_dir):

    <data_dir>/images/                   input images                 (--images_dir)
    <data_dir>/intrinsics.yaml           optional, fixed camera       (--intrinsics_path)
    <data_dir>/camera_poses.yaml         optional, ARKit poses        (--poses_path)
    <data_dir>/sparse/0/*.bin            RESULT: COLMAP binaries only (--output_dir)
    <data_dir>/cache_dir/                everything intermediate      (--cache_dir)
        raw_matches/                     RoMaV2 dense matches, one .npz per pair (survives re-runs)
        pairs.txt, feats-romav2.h5, matches-romav2.h5, sfm/ (COLMAP database, ...)

intrinsics.yaml / camera_poses.yaml come from `sfm/camera_processor.py`. What is estimated depends on the inputs:

    images only                       COLMAP incremental SfM
    + intrinsics                      one shared PINHOLE camera held fixed, COLMAP estimates the poses
    + intrinsics + poses              (default --pose_mode triangulate_ba) triangulate with the given poses, then
                                      global BA with intrinsics fixed and poses + points free

Usage:
    python sfm/reconstruct.py \
        --data_dir local/example \
        --intrinsics_path local/example/intrinsics.yaml \
        --poses_path local/example/camera_poses.yaml \
        --images_dir local/example/images \
        --cache_dir local/example/cache_dir
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger("reconstruct")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def add_hloc_to_path(hloc_dir: str | None) -> None:
    default = Path(__file__).resolve().parents[1] / "third_party" / "Hierarchical-Localization"
    candidates = [hloc_dir, os.environ.get("HLOC_DIR"), str(default)]
    for cand in candidates:
        if cand and (Path(cand) / "hloc").is_dir():
            sys.path.insert(0, str(Path(cand).resolve()))
            break
    try:
        import hloc  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "hloc not found. Clone it with\n"
            "  git clone --depth 1 https://github.com/cvg/Hierarchical-Localization.git "
            "third_party/Hierarchical-Localization\n"
            "or pass --hloc_dir (or set HLOC_DIR)."
        ) from e


def list_images(image_dir: Path) -> list[str]:
    names = [
        p.relative_to(image_dir).as_posix()
        for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    names.sort()
    if not names:
        raise SystemExit(f"No images found in {image_dir}")
    return names


def pairs_exhaustive(names: list[str]) -> list[tuple[str, str]]:
    return list(itertools.combinations(names, 2))


def pairs_sequential(
    names: list[str], window: int, loop: bool
) -> list[tuple[str, str]]:
    n = len(names)
    pairs = set()
    for i in range(n):
        for k in range(1, window + 1):
            j = i + k
            if j < n:
                pairs.add((i, j))
            elif loop and n > window + 1:
                pairs.add((j % n, i))
    return [(names[i], names[j]) for i, j in sorted(pairs)]


def pairs_pose_guided(
    names: list[str],
    poses: dict[str, np.ndarray],
    num_pairs: int,
    max_rot_deg: float,
    min_dist: float = 0.0,
) -> list[tuple[str, str]]:
    """Prior-pose-guided pair selection (replaces NetVLAD retrieval when ARKit poses are available).

    For every query image: (1) coarse filter, keep candidates whose relative rotation (geodesic angle between
    the cam_from_world rotations) is at most `max_rot_deg`; (2) rank the surviving pool by camera-center distance
    (ascending) and keep the top `num_pairs`. Only extrinsics are used, no image content, so repeated or
    low-texture appearance cannot produce geometrically inconsistent pairs, and no global descriptor is extracted.
    The angle of R_i R_j^T does not depend on the camera axis convention, so ARKit/COLMAP conventions both work.
    """
    n = len(names)
    if n < 2:
        return []
    rotations = np.stack([poses[name][:3, :3] for name in names])
    centers = np.stack([-poses[name][:3, :3].T @ poses[name][:3, 3] for name in names])

    # trace(R_i R_j^T) = <R_i, R_j>_F
    cos = (np.einsum("iab,jab->ij", rotations, rotations) - 1.0) / 2.0
    rot_deg = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    dist = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)

    k = min(num_pairs, n - 1)
    pairs: list[tuple[str, str]] = []
    num_fallback = 0
    for i in range(n):
        others = np.arange(n) != i
        pool = np.flatnonzero(others & (rot_deg[i] <= max_rot_deg) & (dist[i] >= min_dist))
        if pool.size:
            chosen = pool[np.argsort(dist[i, pool], kind="stable")[:k]]
        else:
            # nothing passes the filters: link the image to its most similarly oriented frames so it is not isolated
            candidates = np.flatnonzero(others)
            chosen = candidates[np.argsort(rot_deg[i, candidates], kind="stable")[:k]]
            num_fallback += 1
        pairs.extend((names[i], names[int(j)]) for j in chosen)
    if num_fallback:
        logger.warning(
            f"pose-guided pairs: {num_fallback} images had no candidate within {max_rot_deg} deg / "
            f">= {min_dist} m, linked to their {k} closest orientations instead"
        )
    return pairs


def pairs_retrieval(
    args, names: list[str], image_dir: Path, out_dir: Path
) -> list[tuple[str, str]]:
    from hloc import extract_features, pairs_from_retrieval

    conf = extract_features.confs[args.retrieval_model]
    global_feats = extract_features.main(
        conf, image_dir, out_dir, image_list=names
    )
    tmp = out_dir / "pairs-retrieval.txt"
    pairs_from_retrieval.main(
        global_feats, tmp, num_matched=min(args.num_pairs, len(names) - 1)
    )
    return read_pairs(tmp)


def read_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            a, b = line.split()[:2]
            pairs.append((a, b))
    return pairs


def write_pairs(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.write_text("\n".join(f"{a} {b}" for a, b in pairs))


def build_pairs(args, names, image_dir, out_dir, poses=None) -> list[tuple[str, str]]:
    if args.pairs_mode == "pose":
        if poses is None:
            raise SystemExit("--pairs_mode pose requires camera poses (--poses_path)")
        pairs = pairs_pose_guided(names, poses, args.num_pairs, args.pose_max_rot_deg, args.pose_min_dist)
    elif args.pairs_mode == "exhaustive":
        pairs = pairs_exhaustive(names)
    elif args.pairs_mode == "sequential":
        pairs = pairs_sequential(names, args.window, args.loop)
    elif args.pairs_mode == "retrieval":
        pairs = pairs_retrieval(args, names, image_dir, out_dir)
    elif args.pairs_mode == "file":
        if args.pairs_file is None:
            raise SystemExit("--pairs_mode file requires --pairs_file")
        pairs = read_pairs(args.pairs_file)
    else:
        raise SystemExit(f"unknown pairs_mode {args.pairs_mode}")
    # drop duplicates / reversed duplicates, keep order
    seen, uniq = set(), []
    for a, b in pairs:
        if a == b or (a, b) in seen or (b, a) in seen:
            continue
        seen.add((a, b))
        uniq.append((a, b))
    return uniq


class DenseMatcher:
    """Runs RoMaV2 on image pairs and caches raw (pre-quantization) matches, one .npz per pair.

    Keypoints are stored in hloc's convention: pixel centers at integer coordinates,
    i.e. RoMaV2 pixel coordinates (top-left corner origin) minus 0.5.
    """

    def __init__(self, args, image_dir: Path, cache_dir: Path):
        self.args = args
        self.image_dir = image_dir
        self.cache_dir = cache_dir
        self._model = None
        self._sizes: dict[str, tuple[int, int]] = {}

    @property
    def model(self):
        if self._model is None:
            from romav2 import RoMaV2
            from romav2.device import device

            logger.info(f"Loading RoMaV2 (setting={self.args.setting}) on {device}")
            if device.type == "cpu":
                logger.warning("No GPU detected, RoMaV2 will be very slow on CPU.")
            model = RoMaV2()
            model.apply_setting(self.args.setting)
            self._model = model
        return self._model

    def size(self, name: str) -> tuple[int, int]:
        if name not in self._sizes:
            with Image.open(self.image_dir / name) as im:
                self._sizes[name] = im.size
        return self._sizes[name]

    @torch.inference_mode()
    def match_pair(self, n0: str, n1: str):
        model = self.model
        W0, H0 = self.size(n0)
        W1, H1 = self.size(n1)
        preds = model.match(self.image_dir / n0, self.image_dir / n1)
        try:
            matches, overlaps, _, _ = model.sample(preds, self.args.num_matches)
        except RuntimeError as e:
            if "multinomial" not in str(e):
                raise
            logger.warning(f"({n0}, {n1}): not enough overlapping pixels, empty pair")
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), np.zeros(0, np.float32)
        k0, k1 = model.to_pixel_coordinates(matches, H0, W0, H1, W1)
        scores = overlaps.float()
        keep = scores >= self.args.min_score
        k0 = (k0[keep].float() - 0.5).cpu().numpy()
        k1 = (k1[keep].float() - 0.5).cpu().numpy()
        return k0, k1, scores[keep].cpu().numpy()

    def _cache_file(self, n0: str, n1: str) -> Path:
        stem = f"{n0}__{n1}".replace("/", "--")
        return self.cache_dir / f"{stem}.npz"

    def _find_cached(self, n0: str, n1: str):
        """Return (file, swapped) of a cached result for the pair, or (None, False)."""
        f = self._cache_file(n0, n1)
        if f.exists():
            return f, False
        f = self._cache_file(n1, n0)
        if f.exists():
            return f, True
        return None, False

    def run(self, pairs: list[tuple[str, str]]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        todo = [(a, b) for a, b in pairs if self._find_cached(a, b)[0] is None]
        logger.info(f"Dense matching: {len(todo)} new pairs, {len(pairs) - len(todo)} cached")
        for n0, n1 in tqdm(todo, desc="RoMaV2 matching", smoothing=0.1):
            k0, k1, s = self.match_pair(n0, n1)
            f = self._cache_file(n0, n1)
            tmp = f.with_suffix(".tmp.npz")
            np.savez(tmp, keypoints0=k0, keypoints1=k1, scores=s)
            os.replace(tmp, f)

    def export(self, pairs: list[tuple[str, str]], match_path: Path) -> None:
        """Write cached matches into `match_path` using the layout hloc.match_dense produces."""
        from hloc.utils.parsers import names_to_pair

        with h5py.File(match_path, "a") as out:
            for n0, n1 in pairs:
                f, swapped = self._find_cached(n0, n1)
                d = np.load(f)
                k0, k1 = d["keypoints0"], d["keypoints1"]
                if swapped:
                    k0, k1 = k1, k0
                key = names_to_pair(n0, n1)
                if key in out:
                    del out[key]
                grp = out.create_group(key)
                grp.create_dataset("keypoints0", data=k0)
                grp.create_dataset("keypoints1", data=k1)
                grp.create_dataset("scores", data=d["scores"])


def patch_hloc_match_dense(matcher: DenseMatcher) -> None:
    """Swap hloc's learned-matcher stage for RoMaV2, keep its quantization/assignment logic."""
    from hloc import match_dense

    def romav2_match_dense(conf, pairs, image_dir, match_path, existing_refs=()):
        matcher.run(pairs)
        matcher.export(pairs, match_path)

    match_dense.match_dense = romav2_match_dense


def patch_hloc_two_view_geometry() -> None:
    """Make hloc tag the two-view geometries it writes itself as CALIBRATED.

    hloc builds `TwoViewGeometry(inlier_matches=...)` for the skip-verification and known-pose paths. With
    pycolmap >= 4 that leaves config=UNDEFINED, which COLMAP's correspondence graph silently drops, so
    triangulation ends up with 0 points. Pairs verified by `pycolmap.verify_matches` already carry a config.
    """
    from hloc import triangulation

    if isinstance(triangulation.pycolmap, _ConfiguredPycolmap):
        return
    triangulation.pycolmap = _ConfiguredPycolmap(triangulation.pycolmap)


class _ConfiguredPycolmap:
    """Proxy for the pycolmap module that only overrides `TwoViewGeometry`."""

    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        return getattr(self._module, name)

    def TwoViewGeometry(self, *args, **kwargs):
        geometry = self._module.TwoViewGeometry(*args, **kwargs)
        if len(geometry.inlier_matches) > 0 and geometry.config == self._module.TwoViewGeometryConfiguration.UNDEFINED:
            geometry.config = self._module.TwoViewGeometryConfiguration.CALIBRATED
        return geometry


def parse_kv(items, default_options):
    from hloc.triangulation import parse_option_args

    return parse_option_args(items or [], default_options)


def load_intrinsics(path: Path) -> list[float]:
    """Read fx, fy, cx, cy from intrinsics.yaml; every entry must describe the same camera."""
    import yaml

    data = yaml.safe_load(Path(path).read_text())
    params = np.array([entry["params"] for entry in data.values()], dtype=np.float64)
    if params.shape[1] != 4 or not np.allclose(params, params[0]):
        raise SystemExit(f"{path}: expected a single shared [fx, fy, cx, cy] camera")
    return params[0].tolist()


def load_prior_poses(path: Path, names: list[str]) -> dict[str, np.ndarray]:
    """Map image name -> 4x4 cam_from_world from camera_poses.yaml (keys like 'images/000000')."""
    import yaml

    data = yaml.safe_load(Path(path).read_text())["camera_poses"]
    by_stem = {Path(n).stem: n for n in names}
    poses = {}
    for key, entry in data.items():
        name = by_stem.get(Path(key).stem)
        if name is not None:
            poses[name] = np.array(entry["transform_matrix"], dtype=np.float64)
    missing = [n for n in names if n not in poses]
    if missing:
        raise SystemExit(f"{path}: no prior pose for {len(missing)} images, e.g. {missing[:3]}")
    return poses


def run_with_pose_priors(args, pycolmap, sfm_dir, image_dir, pairs_path, feat_path, match_path,
                         names, poses, image_options, mapper_options):
    """hloc's reconstruction.main with ARKit camera centers written to the DB as position priors."""
    from hloc import reconstruction
    from hloc.triangulation import (
        OutputCapture,
        estimation_and_geometric_verification,
        import_features,
        import_matches,
    )

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / "database.db"
    pycolmap.logging.set_log_destination(pycolmap.logging.INFO, sfm_dir / "colmap.LOG.")

    reconstruction.create_empty_db(database)
    reconstruction.import_images(
        image_dir, database, getattr(pycolmap.CameraMode, args.camera_mode), names, image_options
    )
    image_ids = reconstruction.get_image_ids(database)
    cov = np.eye(3) * args.prior_sigma**2
    with pycolmap.Database.open(database) as db:
        import_features(image_ids, db, feat_path)
        import_matches(
            image_ids, db, pairs_path, match_path, args.min_match_score, args.skip_geometric_verification
        )
        for name, image_id in image_ids.items():
            cam_from_world = poses[name]
            center = -cam_from_world[:3, :3].T @ cam_from_world[:3, 3]
            prior = pycolmap.PosePrior()
            prior.corr_data_id = db.read_image(image_id).data_id
            prior.position = center
            prior.position_covariance = cov
            prior.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
            db.write_pose_prior(prior)
    if not args.skip_geometric_verification:
        estimation_and_geometric_verification(database, pairs_path, args.verbose)

    options = {"use_prior_position": True, "use_robust_loss_on_prior_position": True, **mapper_options}
    return reconstruction.run_reconstruction(sfm_dir, database, image_dir, args.verbose, options)


def run_known_pose_triangulation(args, pycolmap, sfm_dir, image_dir, pairs_path, feat_path, match_path,
                                 names, poses, intrinsics):
    """Triangulate with ARKit poses and intrinsics frozen (COLMAP's BA only refines the 3D points)."""
    from hloc import triangulation
    from PIL import Image as PILImage

    with PILImage.open(image_dir / names[0]) as im:
        width, height = im.size
    ref = pycolmap.Reconstruction()
    ref.add_camera_with_trivial_rig(
        pycolmap.Camera(model="PINHOLE", width=width, height=height, params=intrinsics, camera_id=1)
    )
    for image_id, name in enumerate(names, start=1):
        image = pycolmap.Image(name=name, keypoints=np.zeros((0, 2)), camera_id=1, image_id=image_id)
        ref.add_image_with_trivial_frame(image, pycolmap.Rigid3d(poses[name][:3]))
    ref_dir = sfm_dir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref.write(str(ref_dir))
    return triangulation.main(
        sfm_dir, ref_dir, image_dir, pairs_path, feat_path, match_path,
        skip_geometric_verification=args.skip_geometric_verification,
        min_match_score=args.min_match_score, verbose=args.verbose,
    )


def pose_change_vs_prior(rec, poses) -> dict:
    """Camera-center (m) and rotation (deg) deviation of `rec` from the ARKit poses it started from."""
    shifts, angles = [], []
    for image in rec.images.values():
        if not image.has_pose or image.name not in poses:
            continue
        cur = image.cam_from_world().matrix()
        ref = poses[image.name]
        shifts.append(np.linalg.norm((-cur[:3, :3].T @ cur[:3, 3]) - (-ref[:3, :3].T @ ref[:3, 3])))
        cos = (np.trace(cur[:3, :3] @ ref[:3, :3].T) - 1) / 2
        angles.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    shifts, angles = np.array(shifts), np.array(angles)
    return {
        "pos_rms_cm": float(np.sqrt((shifts**2).mean()) * 100),
        "pos_max_cm": float(shifts.max() * 100),
        "rot_mean_deg": float(angles.mean()),
        "rot_max_deg": float(angles.max()),
    }


def refine_poses_global_ba(args, pycolmap, rec, poses):
    """Global BA from the triangulated ARKit-pose model: intrinsics fixed, poses and points free.

    ARKit poses only seed the optimisation (as PriMo does), there is no prior term, so BA can still correct
    ARKit drift. The gauge is anchored on two cameras, which keeps the model in the ARKit frame and metric scale.
    The first round optionally keeps rotations constant so that translations and points settle before the
    rotations are released.
    """
    image_ids = [image_id for image_id, image in rec.images.items() if image.has_pose]
    observations = pycolmap.ObservationManager(rec)

    def solve(constant_rotation: bool):
        config = pycolmap.BundleAdjustmentConfig()
        for image_id in image_ids:
            config.add_image(image_id)
        for camera_id in rec.cameras:
            config.set_constant_cam_intrinsics(camera_id)
        config.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)

        options = pycolmap.BundleAdjustmentOptions()
        options.refine_focal_length = False
        options.refine_principal_point = False
        options.refine_extra_params = False
        options.refine_sensor_from_rig = False
        options.refine_rig_from_world = True
        options.constant_rig_from_world_rotation = constant_rotation
        options.refine_points3D = True
        options.print_summary = False
        options.ceres.loss_function_type = pycolmap.LossFunctionType.HUBER
        options.ceres.loss_function_scale = args.ba_loss_scale
        options.ceres.auto_select_solver_type = False
        solver = options.ceres.solver_options
        solver.linear_solver_type = type(solver.linear_solver_type).SPARSE_SCHUR
        return pycolmap.create_default_bundle_adjuster(options, config, rec).solve()

    def log_state(tag):
        change = pose_change_vs_prior(rec, poses)
        logger.info(
            f"[ba] {tag}: points={rec.num_points3D()} reproj={rec.compute_mean_reprojection_error():.3f}px "
            f"vs ARKit: pos rms={change['pos_rms_cm']:.1f}cm max={change['pos_max_cm']:.1f}cm "
            f"rot mean={change['rot_mean_deg']:.2f}deg max={change['rot_max_deg']:.2f}deg"
        )

    log_state("triangulated")
    if args.ba_min_track_length > 2:
        observations.filter_points3D_with_short_tracks(args.ba_min_track_length)
        log_state(f"dropped tracks shorter than {args.ba_min_track_length}")
    if args.ba_max_points > 0 and rec.num_points3D() > args.ba_max_points:
        rng = np.random.default_rng(args.seed)
        point_ids = np.array(list(rec.points3D.keys()), dtype=np.int64)
        track_lengths = np.array([rec.points3D[int(i)].track.length() for i in point_ids])
        order = np.lexsort((rng.random(len(point_ids)), -track_lengths))
        for point_id in point_ids[order[args.ba_max_points:]]:
            observations.delete_point3D(int(point_id))
        log_state(f"kept the {args.ba_max_points} longest tracks")
    for round_idx in range(args.ba_rounds):
        stages = [False] if args.ba_skip_rotation_stage or round_idx > 0 else [True, False]
        for constant_rotation in stages:
            summary = solve(constant_rotation)
            if not summary.is_solution_usable():
                raise SystemExit(f"[ba] bundle adjustment failed: {summary.brief_report()}")
        num_filtered = observations.filter_all_points3D(args.ba_max_reproj, args.ba_min_tri_angle)
        log_state(f"round {round_idx + 1}/{args.ba_rounds} (filtered {num_filtered} observations)")
    return rec


COLMAP_BINARIES = ("cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin")


def optional_file(path: Path) -> Path | None:
    return path if path.is_file() else None


def write_colmap_binaries(rec, output_dir: Path) -> None:
    """Write the model as COLMAP binaries (replacing a previous result, leaving other files alone)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in COLMAP_BINARIES:
        (output_dir / stale).unlink(missing_ok=True)
    rec.write(str(output_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = parser.add_argument_group("paths")
    g.add_argument("--data_dir", type=Path, required=True,
                   help="scene directory; hosts the inputs and the outputs that are not given explicitly")
    g.add_argument("--images_dir", type=Path, default=None, help="default: <data_dir>/images")
    g.add_argument("--intrinsics_path", type=Path, default=None,
                   help="intrinsics.yaml, one shared PINHOLE camera held fixed (default: <data_dir>/intrinsics.yaml if present)")
    g.add_argument("--poses_path", type=Path, default=None,
                   help="camera_poses.yaml, cam_from_world per image (default: <data_dir>/camera_poses.yaml if present)")
    g.add_argument("--cache_dir", type=Path, default=None,
                   help="intermediate files: RoMaV2 raw matches, pairs, h5 features/matches, COLMAP database "
                        "(default: <data_dir>/cache_dir)")
    g.add_argument("--output_dir", type=Path, default=None,
                   help="receives only the COLMAP binaries (default: <data_dir>/sparse/0)")
    g.add_argument("--hloc_dir", type=str, default=None,
                   help="Hierarchical-Localization checkout (default: $HLOC_DIR or third_party/Hierarchical-Localization)")

    g = parser.add_argument_group("pairs")
    g.add_argument("--pairs_mode", default="auto",
                   choices=["auto", "pose", "exhaustive", "sequential", "retrieval", "file"],
                   help="pose: prior-pose-guided pairs (rotation pre-filter, then nearest camera centers; needs poses, "
                        "skips global-descriptor retrieval). retrieval: global-descriptor retrieval. "
                        "auto: file if --pairs_file is given, else pose if camera poses are available, else exhaustive "
                        "up to --max_exhaustive images, else retrieval")
    g.add_argument("--num_pairs", type=int, default=20, help="pose / retrieval: top-K pairs kept per query image")
    g.add_argument("--pose_max_rot_deg", type=float, default=45.0,
                   help="pose: coarse filter, max relative rotation (deg) between a query and its candidates")
    g.add_argument("--pose_min_dist", type=float, default=0.0,
                   help="pose: min camera-center distance (m) of a candidate, 0 = disabled")
    g.add_argument("--max_exhaustive", type=int, default=80, help="auto: largest image count matched exhaustively")
    g.add_argument("--window", type=int, default=10, help="sequential: match each image with the next N")
    g.add_argument("--loop", action="store_true", help="sequential: wrap around (closed trajectories)")
    g.add_argument("--retrieval_model", default="netvlad", choices=["netvlad", "openibl", "megaloc"])
    g.add_argument("--pairs_file", type=Path, default=None)

    g = parser.add_argument_group("RoMaV2")
    g.add_argument("--setting", default="base", choices=["turbo", "fast", "base", "precise"])
    g.add_argument("--num_matches", type=int, default=5000, help="sampled correspondences per pair")
    g.add_argument("--min_score", type=float, default=0.0, help="drop sampled matches with overlap score below this")
    g.add_argument("--seed", type=int, default=0)

    g = parser.add_argument_group("hloc aggregation")
    g.add_argument("--max_error", type=float, default=2.0, help="px: max distance between a match and its assigned keypoint")
    g.add_argument("--cell_size", type=float, default=4.0, help="px: quantization cell, controls how matches from different pairs share keypoints")
    g.add_argument("--max_kps", type=int, default=0, help="keep top-k keypoints per image by score, 0 = unlimited")

    g = parser.add_argument_group("COLMAP")
    g.add_argument("--camera_mode", default="AUTO", choices=["AUTO", "SINGLE", "PER_FOLDER", "PER_IMAGE"])
    g.add_argument("--camera_model", default=None, help="e.g. SIMPLE_RADIAL, PINHOLE, OPENCV")
    g.add_argument("--skip_geometric_verification", action="store_true")
    g.add_argument("--min_match_score", type=float, default=None)
    g.add_argument("--mapper_options", nargs="*", default=[], help="key=value for pycolmap.IncrementalMapperOptions")
    g.add_argument("--verbose", action="store_true")

    g = parser.add_argument_group("ARKit priors (used when poses are available)")
    g.add_argument("--pose_mode", default=None, choices=["triangulate_ba", "triangulate", "position_prior"],
                   help="triangulate_ba: triangulate with the given poses, then global BA with intrinsics fixed and "
                        "poses + points free; triangulate: poses and intrinsics frozen, triangulation only; "
                        "position_prior: COLMAP estimates the poses, camera centers act as position priors in its BA. "
                        "Default: triangulate_ba with intrinsics, position_prior without")
    g.add_argument("--prior_sigma", type=float, default=0.05, help="m: std of the camera-center prior (pose_mode=position_prior)")
    g.add_argument("--ba_rounds", type=int, default=2, help="triangulate_ba: BA + point filtering rounds")
    g.add_argument("--ba_loss_scale", type=float, default=1.0, help="triangulate_ba: px, Huber loss scale")
    g.add_argument("--ba_min_track_length", type=int, default=3,
                   help="triangulate_ba: drop shorter tracks before BA (they barely constrain poses and cost memory)")
    g.add_argument("--ba_max_points", type=int, default=0,
                   help="triangulate_ba: keep only the N longest tracks (random tie-break) to bound memory, 0 = all")
    g.add_argument("--ba_max_reproj", type=float, default=4.0, help="triangulate_ba: px, filter points above this error after each round")
    g.add_argument("--ba_min_tri_angle", type=float, default=1.5, help="triangulate_ba: deg, filter points below this angle after each round")
    g.add_argument("--ba_skip_rotation_stage", action="store_true",
                   help="triangulate_ba: skip the first BA stage that keeps rotations constant")
    g.add_argument("--only_matching", action="store_true", help="stop after writing feats/matches h5 to the cache")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(name)s %(levelname)s] %(message)s")
    add_hloc_to_path(args.hloc_dir)
    import pycolmap
    from hloc import match_dense, reconstruction

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = args.data_dir.resolve()
    image_dir = (args.images_dir or data_dir / "images").resolve()
    if not image_dir.is_dir():
        raise SystemExit(f"images dir not found: {image_dir}")
    intrinsics_path = args.intrinsics_path or optional_file(data_dir / "intrinsics.yaml")
    poses_path = args.poses_path or optional_file(data_dir / "camera_poses.yaml")
    cache = (args.cache_dir or data_dir / "cache_dir").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    output_dir = (args.output_dir or data_dir / "sparse" / "0").resolve()
    if output_dir == cache or cache in output_dir.parents:
        raise SystemExit("--output_dir must be outside --cache_dir")

    sfm_dir = cache / "sfm"
    raw_dir = cache / "raw_matches"
    feat_path = cache / "feats-romav2.h5"
    match_path = cache / "matches-romav2.h5"
    pairs_path = cache / "pairs.txt"

    names = list_images(image_dir)
    logger.info(f"{len(names)} images in {image_dir}")
    poses = load_prior_poses(poses_path, names) if poses_path else None
    if args.pairs_mode == "auto":
        if args.pairs_file:
            args.pairs_mode = "file"
        elif poses is not None:
            args.pairs_mode = "pose"
        else:
            args.pairs_mode = "exhaustive" if len(names) <= args.max_exhaustive else "retrieval"
        logger.info(f"pairs_mode=auto -> {args.pairs_mode}")
    pairs = build_pairs(args, names, image_dir, cache, poses)
    if not pairs:
        raise SystemExit("no image pairs")
    write_pairs(pairs_path, pairs)
    logger.info(f"{len(pairs)} pairs -> {pairs_path}")

    # The raw RoMaV2 cache survives re-runs; the aggregated files are cheap and always rebuilt
    # so that changing max_error / cell_size / max_kps never requires re-running the network.
    for p in (feat_path, match_path):
        if p.exists():
            p.unlink()

    matcher = DenseMatcher(args, image_dir, raw_dir)
    patch_hloc_match_dense(matcher)
    patch_hloc_two_view_geometry()
    conf = {"output": "matches-romav2", "max_error": args.max_error, "cell_size": args.cell_size}
    match_dense.match_and_assign(
        conf,
        pairs_path,
        image_dir,
        match_path,
        feat_path,
        feature_paths_refs=[],
        max_kps=args.max_kps or None,
        overwrite=True,
    )
    if args.only_matching:
        logger.info(f"features: {feat_path}\nmatches: {match_path}")
        return

    image_options = {}
    if args.camera_model:
        image_options["camera_model"] = args.camera_model
    mapper_options = parse_kv(args.mapper_options, pycolmap.IncrementalMapperOptions())

    intrinsics = None
    if intrinsics_path:
        intrinsics = load_intrinsics(intrinsics_path)
        args.camera_mode = "SINGLE"
        image_options["camera_model"] = "PINHOLE"
        image_options["camera_params"] = ",".join(f"{v:.10g}" for v in intrinsics)
        mapper_options = {
            "ba_refine_focal_length": False,
            "ba_refine_principal_point": False,
            "ba_refine_extra_params": False,
            "mapper": {"abs_pose_refine_focal_length": False, "abs_pose_refine_extra_params": False},
            **mapper_options,
        }
        logger.info(f"fixed intrinsics fx,fy,cx,cy = {intrinsics}")

    if poses is not None and args.pose_mode is None:
        args.pose_mode = "triangulate_ba" if intrinsics is not None else "position_prior"
    if poses is not None:
        logger.info(f"poses: {poses_path}, pose_mode={args.pose_mode}")
    if poses is not None and args.pose_mode in ("triangulate", "triangulate_ba"):
        if intrinsics is None:
            raise SystemExit(f"--pose_mode {args.pose_mode} requires intrinsics (--intrinsics_path)")
        rec = run_known_pose_triangulation(
            args, pycolmap, sfm_dir, image_dir, pairs_path, feat_path, match_path, names, poses, intrinsics
        )
        if args.pose_mode == "triangulate_ba":
            rec = refine_poses_global_ba(args, pycolmap, rec, poses)
    elif poses is not None:
        rec = run_with_pose_priors(
            args, pycolmap, sfm_dir, image_dir, pairs_path, feat_path, match_path,
            names, poses, image_options, mapper_options,
        )
    else:
        rec = reconstruction.main(
            sfm_dir,
            image_dir,
            pairs_path,
            feat_path,
            match_path,
            camera_mode=getattr(pycolmap.CameraMode, args.camera_mode),
            verbose=args.verbose,
            skip_geometric_verification=args.skip_geometric_verification,
            min_match_score=args.min_match_score,
            image_list=names,
            image_options=image_options,
            mapper_options=mapper_options,
        )
    if rec is None:
        raise SystemExit("SfM failed: no model reconstructed (check pairs / num_matches / cell_size)")

    write_colmap_binaries(rec, output_dir)
    logger.info(
        f"Done. registered {rec.num_reg_images()}/{len(names)} images, {rec.num_points3D()} 3D points\n"
        f"  COLMAP bin: {output_dir}\n"
        f"  cache     : {cache}"
    )


if __name__ == "__main__":
    main()
