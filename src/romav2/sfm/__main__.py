"""Run: python -m romav2.sfm --help."""
import argparse
import logging
from pathlib import Path

from .pairs import exhaustive_pairs, read_pairs


def main():
    parser = argparse.ArgumentParser(description='RoMaV2 matching + COLMAP incremental SfM')
    parser.add_argument('--images', type=Path, required=True, help='Root directory containing images')
    parser.add_argument('--output', type=Path, required=True, help='New or empty output directory')
    parser.add_argument('--image-glob', default='**/*', help='Image selection glob relative to --images')
    parser.add_argument('--pairs', type=Path, help='Pairs file: one image name pair per line; default is exhaustive')
    parser.add_argument('--setting', choices=['turbo', 'fast', 'base', 'precise'], default='fast')
    parser.add_argument('--max-matches', type=int, default=5000)
    parser.add_argument('--cell-size', type=float, default=1.0, help='Pixel cell size for merging observations')
    parser.add_argument('--min-matches', type=int, default=15)
    parser.add_argument('--camera-params', nargs=4, type=float, metavar=('FX', 'FY', 'CX', 'CY'),
                        help='Shared COLMAP PINHOLE camera in original-image pixels')
    parser.add_argument('--fix-intrinsics', action='store_true', help='Keep provided calibration fixed during SfM')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('--output must be a new or empty directory')
    if args.fix_intrinsics and args.camera_params is None:
        parser.error('--fix-intrinsics requires --camera-params')
    if args.max_matches <= 0:
        parser.error('--max-matches must be positive')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    from .pairs import IMAGE_EXTENSIONS
    names = sorted(p.relative_to(args.images).as_posix() for p in args.images.glob(args.image_glob)
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if len(names) < 2:
        parser.error('Image selection must contain at least two images')
    pairs = read_pairs(args.pairs, names) if args.pairs else exhaustive_pairs(names)
    logging.info('Selected %d images and %d pairs', len(names), len(pairs))
    from .matching import RomaMatcher
    from .colmap import reconstruct
    matcher = RomaMatcher(args.setting, args.max_matches, args.seed)
    summary = reconstruct(args.images, args.output, names, pairs, matcher, cell_size=args.cell_size,
                          camera_params=args.camera_params, min_matches=args.min_matches,
                          fix_intrinsics=args.fix_intrinsics, seed=args.seed)
    logging.info('Reconstruction summary: %s', summary)


if __name__ == '__main__':
    main()
