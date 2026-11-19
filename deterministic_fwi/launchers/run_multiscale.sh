#!/usr/bin/env bash
# Multi-scale deterministic FWI on the reduced Marmousi.
# Stage 1: residual low-passed at 2.5 Hz (10 iterations)
# Stage 2: 3.5 Hz, warm start from stage 1 (10 more)
# Stage 3: full band, warm start from stage 2 (10 more)
set -euo pipefail

# Parameters
PYTHON="python"
MODEL_PATH="outputs/4d/marmousi4d_expanded_oil_water_white_E2/truth/models/baseline.pth"
BACKGROUND_PATH="outputs/4d/marmousi4d_expanded_oil_water_white_E2/truth/models/background.pth"
DATA_PATH="data/observed_data/freq_5.0Hz_expanded4d_baseline_sequential_sigma_2.6953569406421716_.h5"
OUTPUT_DIR="outputs/4d/deterministic_fwi/reduced_multiscale"

FREQ=5.0
NBL=50
SPACE_ORDER=8
DETH_REC=2
DETH_SRC=2
EDGE_MARGIN_M=0.0
WATER_DEPTH_M=500.0
BG_LINEAR=0
NSRC=100
NREC=400
SIGMA=1.0

CHECKPOINTING=1
NUM_CHECKPOINTS=20
SHOTS_PER_ITER=100
LR=1e-4
WD=0.0
SCALE=0.0
GRAD_CLIP=0.0
MASK_WATER=1
GRAD_SMOOTH=1.5
SAVE_EVERY=1
EXPORT_EVERY=1
SEED=42

LP1=2.5
LP2=3.5
LP3=0.0
ITR1=10
ITR2=20
ITR3=30

PLOT=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
mkdir -p "$OUTPUT_DIR"

run_stage () {
  local lowpass="$1"
  local max_itr="$2"
  local resume="$3"
  "$PYTHON" -m deterministic_fwi.run_inversion \
    --model_path "$MODEL_PATH" \
    --background_path "$BACKGROUND_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --freq "$FREQ" \
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
    --max_itr "$max_itr" \
    --shots_per_iter "$SHOTS_PER_ITER" \
    --lr "$LR" \
    --wd "$WD" \
    --scale "$SCALE" \
    --grad_clip "$GRAD_CLIP" \
    --mask_water "$MASK_WATER" \
    --grad_smooth "$GRAD_SMOOTH" \
    --data_lowpass_hz "$lowpass" \
    --save_every "$SAVE_EVERY" \
    --export_every "$EXPORT_EVERY" \
    --resume "$resume" \
    --retrain 0 \
    --seed "$SEED"
}

run_stage "$LP1" "$ITR1" 0
run_stage "$LP2" "$ITR2" 1
run_stage "$LP3" "$ITR3" 1

if [[ "$PLOT" == "1" ]]; then
  "$PYTHON" -m deterministic_fwi.plot_models \
    --initial_path "$BACKGROUND_PATH" \
    --truth_path "$MODEL_PATH" \
    --models_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR/figures" \
    --watch 0
fi
