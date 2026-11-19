#!/usr/bin/env bash
# Deterministic FWI for the 800-receiver baseline.  VAR=value block, isolated
# output; the final model can be passed as BACKGROUND_PATH to the Deep Prior.
set -euo pipefail

# Parameters
PYTHON="python"
MODEL_PATH="outputs/4d/marmousi4d_expanded_oil_water_dx12p5_E2/truth/models/baseline.pth"
BACKGROUND_PATH="data/velocity_model/marmousi2-model-smooth-linear_freq_12.533333333333333.pth"
DATA_PATH="data/observed_data/freq_12.533333333333333Hz_dx12p5_baseline_sequential_sigma_0.13627556817932762_.h5"
OUTPUT_DIR="outputs/4d/deterministic_fwi/dx12p5_baseline"

FREQ=12.533333333333333
TMAX_MS=8502.256573431108
NBL=150
SPACE_ORDER=8
DETH_REC=5
DETH_SRC=5
EDGE_MARGIN_M=62.5
WATER_DEPTH_M=500.0
BG_LINEAR=0
NSRC=100
NREC=800
SIGMA=0.13627556817932762

CHECKPOINTING=1
NUM_CHECKPOINTS=120
MAX_ITR=20
SHOTS_PER_ITER=100
LR=1e-3
WD=30.0
SCALE=0.0
GRAD_CLIP=0.0
SAVE_EVERY=1
EXPORT_EVERY=1
RESUME=1
RETRAIN=0
SEED=42

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

"$PYTHON" -m deterministic_fwi.run_inversion \
  --model_path "$MODEL_PATH" \
  --background_path "$BACKGROUND_PATH" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --freq "$FREQ" \
  --tmax_ms "$TMAX_MS" \
  --nbl "$NBL" \
  --space_order "$SPACE_ORDER" \
  --deth_rec "$DETH_REC" \
  --deth_src "$DETH_SRC" \
  --edge_margin_m "$EDGE_MARGIN_M" \
  --water_depth_m "$WATER_DEPTH_M" \
  --bg_linear "$BG_LINEAR" \
  --nsrc "$NSRC" \
  --nrec "$NREC" \
  --sigma "$SIGMA" \
  --checkpointing "$CHECKPOINTING" \
  --num_checkpoints "$NUM_CHECKPOINTS" \
  --max_itr "$MAX_ITR" \
  --shots_per_iter "$SHOTS_PER_ITER" \
  --lr "$LR" \
  --wd "$WD" \
  --scale "$SCALE" \
  --grad_clip "$GRAD_CLIP" \
  --save_every "$SAVE_EVERY" \
  --export_every "$EXPORT_EVERY" \
  --resume "$RESUME" \
  --retrain "$RETRAIN" \
  --seed "$SEED"
