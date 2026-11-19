"""Run: python -m romav2.sfm --help."""

import argparse
import logging
from pathlib import Path

from .pairs import discover_images, exhaustive_pairs, read_pairs


def main():
    parser = argparse.ArgumentParser(
        description="RoMaV2 matching and COLMAP reconstruction"
    )
    parser.add_argument(
        "--images", type=Path, required=True, help="Root directory containing images"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New or empty output directory"
    )
    parser.add_argument(
        "--image-glob", default="**/*", help="Image selection glob relative to --images"
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        help="Pairs file: one image name pair per line; default is exhaustive",
    )
    parser.add_argument(
        "--setting", choices=["turbo", "fast", "base", "precise"], default="fast"
    )
    parser.add_argument("--max-matches", type=int, default=5000)
    parser.add_argument(
        "--cell-size",
        type=float,
        default=1.0,
        help="Pixel cell size for merging observations",
    )
    parser.add_argument("--min-matches", type=int, default=15)
    parser.add_argument("--min-match-score", type=float, default=0.0)
    parser.add_argument("--geometric-max-error", type=float, default=4.0)
    parser.add_argument("--geometric-max-trials", type=int, default=20000)
    parser.add_argument("--geometric-min-inlier-ratio", type=float, default=0.1)
    parser.add_argument(
        "--calibration",
        type=Path,
        help="PINHOLE JSON with original image dimensions and params",
    )
    parser.add_argument(
        "--camera-params",
        nargs=4,
        type=float,
        metavar=("FX", "FY", "CX", "CY"),
        help="Shared COLMAP PINHOLE camera in original-image pixels",
    )
    parser.add_argument(
        "--fix-intrinsics",
        action="store_true",
        help="Keep provided calibration fixed during SfM",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--priors",
        type=Path,
        help="JSON containing per-image intrinsics and OpenCV camera-to-world poses",
    )
    parser.add_argument("--intrinsics-prior", action="store_true")
    parser.add_argument("--pose-prior", action="store_true")
    pose_mode = parser.add_mutually_exclusive_group()
    pose_mode.add_argument(
        "--fix-poses",
        action="store_true",
        help="Triangulate with known poses and fixed calibration; requires --priors and --fix-intrinsics",
    )
    pose_mode.add_argument(
        "--init-from-poses",
        action="store_true",
        help="Initialize calibrated cameras and triangulation from full input poses, then refine with soft priors",
    )
    parser.add_argument("--prior-iterations", type=int, default=3)
    parser.add_argument("--camera-model", choices=["PINHOLE", "SIMPLE_RADIAL"])
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument(
        "--match-cache",
        type=Path,
        help="Shared raw match cache; defaults to OUTPUT/match-cache",
    )
    args = parser.parse_args()
    if args.output.exists() and (
        not args.output.is_dir() or any(args.output.iterdir())
    ):
        parser.error("--output must be a new or empty directory")
    if (
        args.fix_intrinsics
        and args.camera_params is None
        and not args.intrinsics_prior
        and args.calibration is None
    ):
        parser.error(
            "--fix-intrinsics requires --camera-params, --calibration, or --intrinsics-prior"
        )
    if args.max_matches <= 0:
        parser.error("--max-matches must be positive")
    if (
        args.intrinsics_prior
        or args.pose_prior
        or args.fix_poses
        or args.init_from_poses
    ) and args.priors is None:
        parser.error("Prior flags require --priors")
    if args.prior_iterations < 1 or args.num_threads < 1:
        parser.error("Prior iterations and thread count must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    names = discover_images(args.images, args.image_glob)
    pairs = read_pairs(args.pairs, names) if args.pairs else exhaustive_pairs(names)
    logging.info("Selected %d images and %d pairs", len(names), len(pairs))
    from .colmap import reconstruct
    from .matching import RomaMatcher
    from .priors import load_priors

    priors = load_priors(args.priors, names) if args.priors else None
    from .calibration import load_calibration

    calibration = load_calibration(args.calibration) if args.calibration else None
    matcher = RomaMatcher(
        args.setting,
        args.max_matches,
        args.seed,
        cache_dir=args.match_cache or args.output / "match-cache",
    )
    summary = reconstruct(
        args.images,
        args.output,
        names,
        pairs,
        matcher,
        cell_size=args.cell_size,
        camera_params=args.camera_params,
        min_matches=args.min_matches,
        fix_intrinsics=args.fix_intrinsics,
        seed=args.seed,
        priors=priors,
        use_intrinsics_prior=args.intrinsics_prior,
        use_pose_prior=args.pose_prior,
        fix_poses=args.fix_poses,
        init_from_poses=args.init_from_poses,
        prior_iterations=args.prior_iterations,
        camera_model=args.camera_model,
        num_threads=args.num_threads,
        calibration=calibration,
        min_match_score=args.min_match_score,
        geometric_max_error=args.geometric_max_error,
        geometric_max_trials=args.geometric_max_trials,
        geometric_min_inlier_ratio=args.geometric_min_inlier_ratio,
    )
    logging.info("Reconstruction summary: %s", summary)


if __name__ == "__main__":
    main()
