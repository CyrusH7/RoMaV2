#!/usr/bin/env bash
set -euo pipefail
if (( $# < 2 )); then
  echo "Usage: $0 OUTPUT_ROOT --images DIR --priors FILE --intrinsics-prior --fix-intrinsics [SfM options]" >&2
  exit 2
fi
output=$1
shift
common=(--joint-prior-ba --match-cache "$output/match-cache" "$@")
for arm in control 2 5 10; do
  mode=(--init-from-poses)
  if [[ $arm != control ]]; then
    mode=(--incremental-prior --registration-iterations "$arm")
  fi
  "${PYTHON_BIN:-python}" -m romav2.sfm "${common[@]}" "${mode[@]}" --output "$output/$arm"
done
