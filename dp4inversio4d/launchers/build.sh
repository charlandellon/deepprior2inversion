#!/usr/bin/env bash
# Build the 4D velocity models and legacy PTH inputs.
set -euo pipefail

# Parameters
PYTHON="python"
CONFIG="dp4inversio4d/configs/marmousi4d_oil_water.yaml"
OUTPUT="outputs/4d/marmousi4d_expanded_oil_water_white_E2"
SEED=""                       # empty keeps the config seed
SCENARIO="E2"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ARGS=(--config "$CONFIG" --output "$OUTPUT")
[[ -n "$SEED" ]] && ARGS+=(--seed "$SEED")
[[ -n "$SCENARIO" ]] && ARGS+=(--scenario "$SCENARIO")
"$PYTHON" -m dp4inversio4d build "${ARGS[@]}"
