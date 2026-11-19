#!/usr/bin/env bash
# 4D posterior statistics for the 32-channel wavelet-correlated 15 dB run.
set -euo pipefail

PYTHON="python"
TRUTH="outputs/4d/marmousi4d_expanded_oil_water_white_E2"
CHECKPOINTS="data/checkpoints"
TAG="expanded4d_s1_c32wc15_g02_b15k"
BURN_IN=15000
STRIDE=10
CLIP_PERCENTILE=99
OUTPUT="$TRUTH/statistics_s1_c32wc15_g02_b15k"
OUTPUT_INDICES="$TRUTH/statistics_s1_c32wc15_g02_b15k_indices"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

"$PYTHON" -m dp4inversio4d stats --truth "$TRUTH" --checkpoints "$CHECKPOINTS" \
  --tag "$TAG" --burn-in "$BURN_IN" --stride "$STRIDE" \
  --clip-percentile "$CLIP_PERCENTILE" --output "$OUTPUT"

"$PYTHON" -m dp4inversio4d.statistical_indices --truth "$TRUTH" \
  --checkpoints "$CHECKPOINTS" --tag "$TAG" --burn-in "$BURN_IN" \
  --stride "$STRIDE" --clip-percentile "$CLIP_PERCENTILE" \
  --output "$OUTPUT_INDICES"
