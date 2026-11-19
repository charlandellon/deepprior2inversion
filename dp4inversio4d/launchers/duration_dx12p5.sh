#!/usr/bin/env bash
# Verify that the acquisition window records the relevant events.
set -euo pipefail

# Parameters
PYTHON="python"
MODEL="data/velocity_model/marmousi2-model-true-large_freq_12.533333333333333.pth"
FACTOR=1.25
NREC=400
NBL=150
OUTPUT="data/images/duration_qc/marmousi_dx12.5"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

"$PYTHON" scripts/modeling/validate_duration.py \
  --model_path "$MODEL" --factor "$FACTOR" --nrec "$NREC" --nbl "$NBL" \
  --shot-fractions 0.0 0.1 0.5 0.9 1.0 --output "$OUTPUT"
