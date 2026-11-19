#!/usr/bin/env bash
# Clean/noisy central gathers for the 12.5 m experiment.
set -euo pipefail

PYTHON="python"
TRUTH="outputs/4d/marmousi4d_expanded_oil_water_dx12p5_E2"
MODELS="$TRUTH/truth/models"
SOURCE_MODE="simult"
DATA_DIR="data/observed_data"
TAG="dx12p5"
NSRC=100
NREC=800
NSIMSRC=100
FREQ=12.533333333333333
NBL=150
CHECKPOINTING=1
NUM_CHECKPOINTS=0
DETH_REC=5
DETH_SRC=5
EDGE_MARGIN_M=62.5
WATER_DEPTH_M=500.0
# 1.75 * diagonal/mean-vp; duration QC showed ~99.9% energy capture.
TMAX_MS=8502.256573431108

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "$MODELS/baseline.pth" ]]; then
  echo "missing $MODELS/baseline.pth; run build_dx12p5.sh first" >&2
  exit 1
fi

"$PYTHON" -m dp4inversio4d gathers --truth "$TRUTH" --source-mode "$SOURCE_MODE" \
  --data-dir "$DATA_DIR" --tag "$TAG" --nsrc "$NSRC" --nrec "$NREC" \
  --nsimsrc "$NSIMSRC" --freq "$FREQ" --nbl "$NBL" --tmax-ms "$TMAX_MS" \
  --deth-rec "$DETH_REC" --deth-src "$DETH_SRC" \
  --edge-margin-m "$EDGE_MARGIN_M" --water-depth-m "$WATER_DEPTH_M" \
  --checkpointing "$CHECKPOINTING" --num-checkpoints "$NUM_CHECKPOINTS"
