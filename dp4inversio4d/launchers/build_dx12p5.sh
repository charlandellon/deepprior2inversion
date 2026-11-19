#!/usr/bin/env bash
# Build extended oil/water models on the 12.5 m grid.
set -euo pipefail

PYTHON="python"
CONFIG="dp4inversio4d/configs/marmousi4d_oil_water_dx12p5.yaml"
OUTPUT="outputs/4d/marmousi4d_expanded_oil_water_dx12p5_E2"
SCENARIO="E2"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

"$PYTHON" -m dp4inversio4d build --config "$CONFIG" --output "$OUTPUT" \
  --scenario "$SCENARIO"
