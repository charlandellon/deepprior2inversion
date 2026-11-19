#!/usr/bin/env bash
# Continuously plot every exported deterministic-FWI velocity model.
set -euo pipefail

# Parameters
PYTHON="python"
INITIAL_PATH="data/velocity_model/marmousi2-model-smooth-linear_freq_12.533333333333333.pth"
TRUTH_PATH="outputs/4d/marmousi4d_expanded_oil_water_dx12p5_E2/truth/models/baseline.pth"
MODELS_DIR="outputs/4d/deterministic_fwi/dx12p5_baseline"
OUTPUT_DIR="outputs/4d/deterministic_fwi/dx12p5_baseline/figures"
WATCH=1
INTERVAL=30

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

"$PYTHON" -m deterministic_fwi.plot_models \
  --initial_path "$INITIAL_PATH" \
  --truth_path "$TRUTH_PATH" \
  --models_dir "$MODELS_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --watch "$WATCH" \
  --interval "$INTERVAL"
