#!/usr/bin/env bash
# Reservoir-focused statistical indices for a 4D run.
set -euo pipefail

PYTHON="python"
TRUTH="outputs/4d/marmousi4d_expanded_oil_water_white_E2"
CHECKPOINTS="data/checkpoints"
TAG="expanded4d_s1"
SURVEYS="baseline monitor1 monitor2"
BURN_IN=17500
STRIDE=10
CLIP_PERCENTILE=99
SAMPLE_QUANTILES="10 50 90"
OUTPUT="$TRUTH/statistics_s1_indices"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

"$PYTHON" -m dp4inversio4d.statistical_indices \
  --truth "$TRUTH" --checkpoints "$CHECKPOINTS" --tag "$TAG" \
  --surveys $SURVEYS --burn-in "$BURN_IN" --stride "$STRIDE" \
  --clip-percentile "$CLIP_PERCENTILE" \
  --sample-quantiles $SAMPLE_QUANTILES --output "$OUTPUT"
