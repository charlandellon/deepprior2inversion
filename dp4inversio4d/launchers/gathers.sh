#!/usr/bin/env bash
# Plot the central clean/noisy gather for each survey after data generation.
set -euo pipefail

# Parameters
PYTHON="python"
TRUTH="outputs/4d/marmousi4d_expanded_oil_water_white_E2"
SOURCE_MODE="simult"          # sequential | simult
DATA_DIR="data/observed_data"
TAG="expanded4d"
NSRC=100
NREC=400
NSIMSRC=100
FREQ=5.0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

"$PYTHON" -m dp4inversio4d gathers --truth "$TRUTH" --source-mode "$SOURCE_MODE" \
  --data-dir "$DATA_DIR" --tag "$TAG" --nsrc "$NSRC" --nrec "$NREC" \
  --nsimsrc "$NSIMSRC" --freq "$FREQ"
