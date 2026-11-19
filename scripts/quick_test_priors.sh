#!/usr/bin/env bash
# Activate romav2 first. All arms reuse identical images/pairs/raw matches.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-$REPO_DIR/data/7-scenes/office}"
RUN_DIR="${RUN_DIR:-$REPO_DIR/runs/office_prior_ablation_$(date +%Y%m%d_%H%M%S)}"
COUNT="${COUNT:-100}"
THREADS="${THREADS:-4}"
CALIBRATION="${CALIBRATION:-$REPO_DIR/src/romav2/sfm/configs/office.json}"
calibration_flags=(--calibration "$CALIBRATION" --rig-calibration "${RIG_CALIBRATION:-$REPO_DIR/src/romav2/sfm/configs/7scenes_rig.json}")
if [[ -v FX || -v FY || -v CX || -v CY ]]; then
  : "${FX:?Set all of FX,FY,CX,CY to override calibration}"
  : "${FY:?Set all of FX,FY,CX,CY to override calibration}"
  : "${CX:?Set all of FX,FY,CX,CY to override calibration}"
  : "${CY:?Set all of FX,FY,CX,CY to override calibration}"
  calibration_flags+=(--intrinsics "$FX" "$FY" "$CX" "$CY")
fi
export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
"$PYTHON_BIN" -m romav2.sfm.ablation prepare --dataset "$DATASET" \
  --sequence "${SEQUENCE:-seq-01}" --output "$RUN_DIR" --count "$COUNT" \
  --window "${PAIR_WINDOW:-5}" --seed "${SEED:-0}" \
  --noise-m "${PRIOR_NOISE_M:-0.02}" --noise-deg "${PRIOR_NOISE_DEG:-2}" \
  --position-sigma "${POSITION_SIGMA:-0.05}" --rotation-sigma-deg "${ROTATION_SIGMA_DEG:-3}" \
  "${calibration_flags[@]}"
if [[ "${PREPARE_ONLY:-0}" == 1 ]]; then
  echo "Prepared experiment: $RUN_DIR"
  exit 0
fi
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "Activate a GPU before running RoMaV2 ablations"'
failed=0
arms=(baseline intrinsics extrinsics both)
eval_flags=()
if [[ "${FIXED_POSE_CONTROL:-0}" == 1 ]]; then
  arms+=(fixed_poses)
  eval_flags+=(--with-fixed-poses)
fi
if [[ "${PRIOR_INIT_CONTROL:-0}" == 1 ]]; then
  arms+=(prior_init)
  eval_flags+=(--with-prior-init)
fi
for arm in "${arms[@]}"; do
  flags=()
  if [[ "$arm" == intrinsics || "$arm" == both ]]; then flags+=(--intrinsics-prior --fix-intrinsics); fi
  if [[ "$arm" == extrinsics || "$arm" == both ]]; then flags+=(--pose-prior); fi
  if [[ "$arm" == fixed_poses ]]; then flags+=(--intrinsics-prior --fix-intrinsics --fix-poses); fi
  if [[ "$arm" == prior_init ]]; then flags+=(--intrinsics-prior --fix-intrinsics --init-from-poses); fi
  echo "Running $arm; log: $RUN_DIR/$arm.log"
  if "$PYTHON_BIN" -u -m romav2.sfm --images "$RUN_DIR/images" --output "$RUN_DIR/$arm" \
    --pairs "$RUN_DIR/pairs.txt" --priors "$RUN_DIR/priors.json" \
    --camera-model PINHOLE --setting "${SETTING:-fast}" --max-matches "${MAX_MATCHES:-2000}" \
    --min-match-score "${MIN_MATCH_SCORE:-0}" --cell-size "${CELL_SIZE:-2}" --seed "${SEED:-0}" --num-threads "$THREADS" \
    --prior-iterations "${PRIOR_ITERATIONS:-3}" --match-cache "$RUN_DIR/match-cache" \
    "${flags[@]}" > "$RUN_DIR/$arm.log" 2>&1; then
    echo "Finished $arm"
  else
    echo "FAILED $arm; inspect $RUN_DIR/$arm.log" >&2
    failed=1
  fi
done
if ! "$PYTHON_BIN" -m romav2.sfm.ablation evaluate --output "$RUN_DIR" "${eval_flags[@]}" | tee "$RUN_DIR/evaluation.log"; then
  failed=1
fi
echo "Results: $RUN_DIR/comparison.csv"
exit "$failed"
