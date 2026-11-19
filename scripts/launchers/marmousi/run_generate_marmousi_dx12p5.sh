#!/usr/bin/env bash
# Generate the 12.5 m Marmousi models by only changing f0.
# calc_dx_dt gives dx = vmax/(30*f0); for vmax=4700 m/s -> f0=12.5333 Hz.
set -euo pipefail

# Parameters
PYTHON="python"
F0="12.533333333333333"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

"$PYTHON" scripts/velocity_models/gerando_marmousi.py --f0 "$F0"
