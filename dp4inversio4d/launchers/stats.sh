#!/usr/bin/env bash
# Quantify 4D posterior statistics after all three inversions finish.
set -euo pipefail

# Parameters
PYTHON="python"
TRUTH="outputs/4d/marmousi4d_expanded_oil_water_white_E2"
CHECKPOINTS="data/checkpoints"
TAG="expanded4d"
BURN_IN=17500
STRIDE=10
CLIP_PERCENTILE=99
OUTPUT="$TRUTH/statistics"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

"$PYTHON" -m dp4inversio4d stats --truth "$TRUTH" --checkpoints "$CHECKPOINTS" \
  --tag "$TAG" --burn-in "$BURN_IN" --stride "$STRIDE" \
  --clip-percentile "$CLIP_PERCENTILE" --output "$OUTPUT"
