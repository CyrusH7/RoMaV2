"""RoMaV2 quick start: match two images, sample correspondences, estimate F, save a visualization.

Usage (from repo root):
    python demo/quickstart.py
    python demo/quickstart.py --im_A_path a.jpg --im_B_path b.jpg --setting fast --num_matches 2000
"""

import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from romav2 import RoMaV2
from romav2.device import device


def draw_matches(im_A, im_B, kptsA, kptsB, inlier_mask=None, max_draw=200, seed=0):
    W_A, H_A = im_A.size
    W_B, H_B = im_B.size
    canvas = Image.new("RGB", (W_A + W_B, max(H_A, H_B)), (255, 255, 255))
    canvas.paste(im_A, (0, 0))
    canvas.paste(im_B, (W_A, 0))
    draw = ImageDraw.Draw(canvas)

    idx = np.arange(len(kptsA))
    if inlier_mask is not None:
        idx = idx[inlier_mask]
    rng = np.random.default_rng(seed)
    if len(idx) > max_draw:
        idx = rng.choice(idx, max_draw, replace=False)
    for i in idx:
        xa, ya = kptsA[i]
        xb, yb = kptsB[i]
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        draw.line([(xa, ya), (xb + W_A, yb)], fill=color, width=1)
        draw.ellipse([xa - 2, ya - 2, xa + 2, ya + 2], outline=color)
        draw.ellipse([xb + W_A - 2, yb - 2, xb + W_A + 2, yb + 2], outline=color)
    return canvas


def main():
    parser = ArgumentParser()
    parser.add_argument("--im_A_path", default="assets/toronto_A.jpg")
    parser.add_argument("--im_B_path", default="assets/toronto_B.jpg")
    parser.add_argument(
        "--setting",
        default="base",
        choices=["turbo", "fast", "base", "precise"],
        help="turbo/fast are fastest; precise is the most accurate and uses the most memory",
    )
    parser.add_argument("--num_matches", type=int, default=5000)
    parser.add_argument("--save_path", default="demo/quickstart_matches.jpg")
    args = parser.parse_args()

    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    t0 = time.time()
    model = RoMaV2()
    model.apply_setting(args.setting)
    print(f"model loaded in {time.time() - t0:.1f}s, setting={args.setting}")

    im_A = Image.open(args.im_A_path).convert("RGB")
    im_B = Image.open(args.im_B_path).convert("RGB")
    W_A, H_A = im_A.size
    W_B, H_B = im_B.size
    print(f"image A: {W_A}x{H_A}, image B: {W_B}x{H_B}")

    # The first call includes CUDA warmup, so time the second one.
    model.match(args.im_A_path, args.im_B_path)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    preds = model.match(args.im_A_path, args.im_B_path)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"match: {time.time() - t0:.3f}s")
    print(
        "warp_AB:", tuple(preds["warp_AB"].shape),
        "overlap_AB:", tuple(preds["overlap_AB"].shape),
    )

    matches, overlaps, precision_AB, precision_BA = model.sample(preds, args.num_matches)
    kptsA, kptsB = model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
    kptsA, kptsB = kptsA.cpu().numpy(), kptsB.cpu().numpy()
    print(f"sampled {len(kptsA)} matches, mean overlap={overlaps.mean().item():.3f}")

    assert torch.isfinite(matches).all(), "matches contain NaN/Inf"
    assert len(kptsA) > 0, "no matches sampled"

    inlier_mask = None
    try:
        import cv2

        F, mask = cv2.findFundamentalMat(
            kptsA,
            kptsB,
            ransacReprojThreshold=0.2,
            method=cv2.USAC_MAGSAC,
            confidence=0.999999,
            maxIters=10000,
        )
        if F is None:
            print("fundamental matrix estimation failed")
        else:
            inlier_mask = mask.ravel().astype(bool)
            print(f"fundamental matrix inliers: {inlier_mask.sum()}/{len(inlier_mask)}")
            print("F =\n", F)
    except ImportError:
        print("opencv not installed, skip fundamental matrix (pip install opencv-python-headless)")

    vis = draw_matches(im_A, im_B, kptsA, kptsB, inlier_mask)
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    vis.save(args.save_path)
    print(f"saved visualization to {args.save_path}")
    print("OK")


if __name__ == "__main__":
    main()
