#!/usr/bin/env bash
# 4D posterior statistics for the 12.5 m experiment.
set -euo pipefail

PYTHON="python"
TRUTH="outputs/4d/marmousi4d_expanded_oil_water_dx12p5_E2"
CHECKPOINTS="data/checkpoints"
TAG="dx12p5"
BURN_IN=17500
STRIDE=10
CLIP_PERCENTILE=99
OUTPUT="$TRUTH/statistics"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

"$PYTHON" -m dp4inversio4d stats --truth "$TRUTH" --checkpoints "$CHECKPOINTS" \
  --tag "$TAG" --burn-in "$BURN_IN" --stride "$STRIDE" \
  --clip-percentile "$CLIP_PERCENTILE" --output "$OUTPUT"
