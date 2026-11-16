<p align="center">
  <h1 align="center"> <ins>RoMa v2</ins> 🤖: Harder Better Faster Denser Feature Matching
  <h2 align="center">
    <a href="https://scholar.google.com/citations?user=Ul-vMR0AAAAJ">Johan Edstedt</a>
    ·
    <a href="https://scholar.google.com/citations?user=-vJPE04AAAAJ">David Nordström</a>
    ·
    <a href="https://scholar.google.com/citations?user=mvY4rdIAAAAJ">Yushan Zhang</a>
    ·
    <a href="https://scholar.google.com/citations?user=FUE3Wd0AAAAJ">Georg Bökman</a>
    ·
    <a href="https://scholar.google.com/citations?user=dsEPAvUAAAAJ">Jonathan Astermark</a>
    ·
    <a href="https://scholar.google.com/citations?user=vHeD0TYAAAAJ">Viktor Larsson</a>
    ·
    <a href="https://scholar.google.com/citations?user=9j-6i_oAAAAJ&hl">Anders Heyden</a>
    ·
    <a href="https://scholar.google.com/citations?user=P_w6UgMAAAAJ&hl">Fredrik Kahl</a>
    ·
    <a href="https://scholar.google.com/citations?user=6WRQpCQAAAAJ">Mårten Wadenbäck</a>
    ·
    <a href="https://scholar.google.com/citations?user=lkWfR08AAAAJ">Michael Felsberg</a>
  </p>
  <h2 align="center"><p>
    <a href="https://arxiv.org/abs/2511.15706" align="center">Paper</a> | 
    <a href="TBD" align="center">Project Page</a>
  </p></h2>
  <div align="center"></div>
</p>
<br/>
<p align="center">
    <img src="assets/qualitative.png" alt="example" width=80%>
</p>

## How to Use
```python
from romav2 import RoMaV2

# load pretrained model
model = RoMaV2()
# Match densely for any image-like pair of inputs
preds = model.match(img_A_path, img_B_path)

# you can also run the forward method directly as 
# preds = model(img_A, img_B)

# Sample 5000 matches for estimation
matches, overlaps, precision_AB, precision_BA = model.sample(preds, 5000)

# Convert to pixel coordinates (RoMaV2 produces matches in [-1,1]x[-1,1])
kptsA, kptsB = model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

# Find a fundamental matrix (or anything else of interest)
F, mask = cv2.findFundamentalMat(
    kptsA.cpu().numpy(), kptsB.cpu().numpy(), ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000
)
```
We additionally provide two demos in the [demos folder](demo), which might help in understanding.


## Setup/Install
In your python environment (tested on Linux python 3.12), run:
```bash
uv pip install -e .
```
or 
```bash
uv sync
```

## Benchmarks
If you do not already have MegaDepth and ScanNet, you can the following to download them:
```bash
source scripts/eval_prep.sh
```
### Mega-1500
```bash
uv run tests/test_mega1500.py
```
### ScanNet-1500
```bash
uv run tests/test_scannet1500.py
```
### Expected Results
Experiments on ScanNet-1500 and MegaDepth-1500 are provided in the [tests folder](tests).
Running these gave me `ScanNet-1500: [34.0, 56.5, 73.9]`, and `Mega-1500: [62.8, 76,8, 86.5]`, which are similar to the results of the paper.


## Fused local correlation kernel
Include the `--extra fused-local-corr` flag as:
```bash
uv sync --extra fused-local-corr
```
or 
```bash
uv pip install romav2[fused-local-corr]
```
or
```bash
uv add romav2[fused-local-corr]
```

## Settings
By twiddling with some different settings you may reach better results on your task of interest.
Some important ones, which we enable setting to some reasonable defaults through `model.apply_setting`, are:

`model.H_lr, model.W_lr`: height and width for the image pair.

`model.H_hr, model.W_hr`: height and width for a high resolution version of the image pair (used for upsampling as in RoMa)

`model.bidirectional`: Useful for getting more diverse matches, and for estimating the covariance matrix in both directions.

`model.threshold`: Value between [0,1]. Used to set overlap prediction above it to 1. Useful for Mega1500.

`model.balanced_sampling`: Diverse sampling, same as RoMa. Typically helps to get better RANSAC estimates.

## License
All our code except DINOv3 is MIT license.
DINOv3 has a custom license, see [DINOv3](https://github.com/facebookresearch/dinov3/tree/main?tab=License-1-ov-file#readme).

## Acknowledgement
Our codebase builds mainly on the code in [RoMa](https://github.com/Parskatt/RoMa).
We were additionally inspired by [UFM](https://github.com/UniFlowMatch/UFM) and [MapAnything](https://github.com/facebookresearch/map-anything), particularly for the datasets used to train the models.

## BibTeX
If you find our models useful, please consider citing our paper!
```
@article{edstedt2025romav2,
  title={{RoMa v2: Harder Better Faster Denser Feature Matching}},
  author={Johan Edstedt, David Nordström, Yushan Zhang, Georg Bökman, Jonathan Astermark, Viktor Larsson, Anders Heyden, Fredrik Kahl, Mårten Wadenbäck, Michael Felsberg},
  journal={arXiv preprint arXiv:2511.15706},
  year={2025}
}
```

## RoMaV2 + COLMAP SfM (experimental)

An optional pipeline in `romav2.sfm` matches image pairs with RoMaV2, merges
pairwise observations into shared per-image keypoints, writes a COLMAP database,
performs geometric verification, removes verified match edges that would create
tracks with multiple observations in one image, and runs COLMAP's incremental
mapper and bundle adjustment. This is the **image-only baseline**: it does not yet read pose or
depth priors, perform pose-guided pair selection or prior-guided refinement, or
recover metric scale. A monocular reconstruction is defined only up to similarity.

Install the optional backend in the existing environment:

```bash
conda activate romav2
uv pip install -e '.[sfm]'
```

Run on a small image collection (the output directory must be empty):

```bash
python -m romav2.sfm --images /path/to/images --output /path/to/new-output \
    --setting fast
```

By default all image pairs are matched, so cost grows quadratically. For larger
collections provide `--pairs /path/to/pairs.txt`: each line contains two image
paths relative to `--images`, separated by whitespace. Reversed duplicates are
ignored. Image names must not contain whitespace. This file is the integration
point for a future pose-based pair selector. Use `--image-glob` to filter input,
e.g. `--image-glob '**/*.color.png'` to exclude depth PNGs in 7-Scenes.

For a shared calibrated PINHOLE camera, supply original-image pixel intrinsics:

```bash
python -m romav2.sfm --images /path/to/images --output /path/to/new-output \
    --camera-params FX FY CX CY --fix-intrinsics
```

All selected images must have the same size for this shared camera. Without
`--fix-intrinsics`, COLMAP may optimize focal length and distortion where
applicable. Without `--camera-params`, COLMAP initializes cameras from image
metadata or its default heuristic. Images should be undistorted if using the
PINHOLE model. COLMAP pixel coordinates place the first pixel center at (0.5,
0.5). After geometric verification, conflict filtering prefers higher-confidence matches;
`summary.json` records the number of rejected conflicting edges.
`--cell-size` (default 1 pixel) controls observation merging: increasing
it can create more tracks but may merge distinct features and degrade accuracy.

Outputs are `database.db`, `pairs.txt`, `summary.json`, and COLMAP binary models
under `sparse/<model-id>/`. There may be multiple disconnected models; inspect
`summary.json` for registered-image and point counts. No-model runs fail with a
nonzero exit code. PyCOLMAP 3.13 is required for this integration. The first
RoMaV2 run downloads model weights and DINOv3 code, and GPU execution has to be
validated with the installed PyTorch/CUDA build and available hardware. CPU
sampling and local correlation are supported, but full CPU model inference can be
very slow. Pairs with no valid correspondences are skipped by COLMAP; sampling
returns at most the number of available valid correspondences. See
`python -m romav2.sfm --help` for all options.

For a CPU-only diagnostic of the COLMAP backend on 7-Scenes color frames, install
`opencv-python-headless` and run `python scripts/check_colmap_sift.py --images
/path/to/seq --output /path/to/new-output`. This uses SIFT, not RoMaV2, and does
not validate the learned matcher. 7-Scenes RGB calibration is approximate; use
appropriate RGB intrinsics for the data and intended evaluation rather than
assuming that the dataset's depth-camera calibration applies to RGB.
