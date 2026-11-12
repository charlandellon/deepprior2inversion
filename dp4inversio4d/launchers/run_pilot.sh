#!/usr/bin/env bash
# Short controlled continuation of the baseline chain for sampler pilots.
set -euo pipefail

EXPERIMENT="${1:?usage: run_pilot.sh experiment scale lr max_itr source_checkpoint}"
SCALE="${2:?scale}"
LR="${3:?learning rate}"
MAX_ITR="${4:?max iteration}"
SOURCE="${5:?source checkpoint}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TRUTH="outputs/4d/marmousi4d_expanded_oil_water_white_E2"
MODELS="$TRUTH/truth/models"
DEST="data/checkpoints/4d/$EXPERIMENT/$EXPERIMENT"
mkdir -p "$DEST"
cp "$SOURCE" "$DEST/checkpoint.pth"

SIGMA="$(python -c "import json;print(repr(json.load(open('$MODELS/sigma_baseline.json'))['sigma']))")"

python scripts/inversion/run_imaging.py --phase sample --cuda 1 \
  --experiment "$EXPERIMENT" \
  --file_figs "4d/$EXPERIMENT" \
  --file_simut_data expanded4d_baseline_simult.h5 \
  --file_seq_data expanded4d_baseline_sequential.h5 \
  --vel_model marmousi --model_path "$MODELS/baseline.pth" \
  --background_path "$MODELS/background.pth" \
  --noise_type white_gaussian --sigma "$SIGMA" --seed 42 \
  --max_itr "$MAX_ITR" --lr "$LR" --lr_final "$LR" --wd 30.0 \
  --gamma -0.3333 --scale "$SCALE" --scale_final "$SCALE" \
  --nsrc 100 --nrec 400 --nsimsrc 100 --freq 5.0 --simult 1 \
  --run_all 0 --retrain 0 --bg_linear 1 --short_output \
  --upsample_scale 1 --coarse_output_scale 1 \
  --ch_encoding 16 16 16 --ch_decoding 16 16 16 --ch_skip 16 16 16
