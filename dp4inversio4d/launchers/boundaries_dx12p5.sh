#!/usr/bin/env bash
# Absorbing-boundary QC on the 12.5 m grid, without saving the full wavefield.
set -euo pipefail

# Parameters
PYTHON="python"
MODEL="data/velocity_model/marmousi2-model-true-large_freq_12.533333333333333.pth"
NBL_LIST="50 100 150 200 250 300"
NREC=400
OUTPUT="data/images/boundary_qc/marmousi_dx12.5"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

"$PYTHON" scripts/modeling/test_absorber_boundaries.py \
  --model_path "$MODEL" --nbl $NBL_LIST --nrec "$NREC" \
  --shot-fractions 0.0 0.5 1.0 --output "$OUTPUT"
