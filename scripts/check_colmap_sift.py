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


def sift_matcher(images, names, max_features=6000, downsample=1):
    sift = cv2.SIFT_create(nfeatures=max_features)
    features = {}
    for name in names:
        path = (images / name).resolve()
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot read image: {name}")
        h, w = image.shape
        if downsample > 1:
            image = cv2.resize(
                image,
                (max(1, w // downsample), max(1, h // downsample)),
                interpolation=cv2.INTER_AREA,
            )
        keypoints, descriptors = sift.detectAndCompute(image, None)
        xy = np.asarray([key.pt for key in keypoints], dtype=float).reshape(-1, 2)
        # Convert to COLMAP pixel centers at the original RGB resolution.
        features[path] = (
            (xy + 0.5) * [w / image.shape[1], h / image.shape[0]],
            descriptors,
        )

    def match(a, b):
        xy_a, da = features[a.resolve()]
        xy_b, db = features[b.resolve()]
        if da is None or db is None or len(db) < 2:
            return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
        neighbors = cv2.BFMatcher().knnMatch(da, db, k=2)
        good = [m for m, n in neighbors if m.distance < 0.75 * n.distance]
        return (
            xy_a[[m.queryIdx for m in good]],
            xy_b[[m.trainIdx for m in good]],
            np.asarray([1 / (1 + m.distance) for m in good]),
        )

    return match


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
    match = sift_matcher(args.images, names)
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
