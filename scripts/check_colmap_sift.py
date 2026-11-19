"""CPU diagnostic for the COLMAP backend, without RoMaV2 model weights.

This is a backend check only: SIFT matches do not validate RoMaV2 inference.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from romav2.sfm.calibration import load_calibration
from romav2.sfm.colmap import reconstruct
from romav2.sfm.pairs import IMAGE_EXTENSIONS, exhaustive_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-glob", default="**/*.color.png")
    parser.add_argument(
        "--camera-params", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY")
    )
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    names = sorted(
        p.relative_to(args.images).as_posix()
        for p in args.images.glob(args.image_glob)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(names) < 2:
        parser.error("At least two images are required")
    sift = cv2.SIFT_create(nfeatures=6000)
    features = {}
    for name in names:
        image = cv2.imread(str(args.images / name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot read image: {name}")
        keypoints, descriptors = sift.detectAndCompute(image, None)
        features[(args.images / name).resolve()] = (keypoints, descriptors)

    def match(a, b):
        ka, da = features[a.resolve()]
        kb, db = features[b.resolve()]
        if da is None or db is None or len(db) < 2:
            return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
        neighbors = cv2.BFMatcher().knnMatch(da, db, k=2)
        good = [m for m, n in neighbors if m.distance < 0.75 * n.distance]
        # OpenCV uses a top-left pixel center of (0, 0); COLMAP uses (0.5, 0.5).
        xy_a = (
            np.asarray([ka[m.queryIdx].pt for m in good], dtype=float).reshape(-1, 2)
            + 0.5
        )
        xy_b = (
            np.asarray([kb[m.trainIdx].pt for m in good], dtype=float).reshape(-1, 2)
            + 0.5
        )
        return xy_a, xy_b, np.asarray([1 / (1 + m.distance) for m in good])

    summary = reconstruct(
        args.images,
        args.output,
        names,
        exhaustive_pairs(names),
        match,
        camera_params=args.camera_params,
        calibration=load_calibration(args.calibration) if args.calibration else None,
    )
    print(summary)


if __name__ == "__main__":
    main()
