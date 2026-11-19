#!/usr/bin/env bash
# One independent 4D chain on the 12.5 m grid with wavelet-correlated noise.
set -euo pipefail

SURVEY="${1:?usage: run_dx12p5.sh baseline|monitor1|monitor2}"

# Paths
PYTHON="python"
TRUTH="outputs/4d/marmousi4d_expanded_oil_water_dx12p5_E2"
MODELS="$TRUTH/truth/models"

# Acquisition
SOURCE_MODE="simult"
NSRC=100
NREC=800
NSIMSRC=100
FREQ=12.533333333333333
NOISE_TYPE="wavelet_correlated"
TARGET_SNR_DB=6.0
NBL=150
CHECKPOINTING=1
DATA_CHECKPOINTING=1
NUM_CHECKPOINTS=120
DETH_REC=5
DETH_SRC=5
EDGE_MARGIN_M=62.5
NP_KERNEL=120
WATER_DEPTH_M=500.0
BG_SIGMA=125.33333333333333
# 1.75 * diagonal/mean-vp; duration QC showed ~99.9% energy capture.
TMAX_MS=8502.256573431108
RUN_ALL=0
RETRAIN=0
BG_LINEAR=1
SHORT_OUTPUT=1
MAX_RETRIES=5
RETRY_WAIT_S=30

# Inversion
# Full chain. Measured ~21.4 s/iteration with 120 checkpoints
# (stable ~1.5 GiB) on the current 12.5 m grid.
MAX_ITR=25000
CUDA=1
LR=0.001
LR_FINAL=0.0001
WD=30.0
GAMMA=-0.3333
SCALE=1.0
UPSCALE=1
COARSE_OUTPUT_SCALE=1
CH_ENCODING="32 32 32"
CH_DECODING="32 32 32"
CH_SKIP="32 32 32"

# Experiment naming
EXPERIMENT_PREFIX="dx12p5_baseline_800rec_c32"
FILE_FIGS_PREFIX="4d/dx12p5_baseline_800rec_c32"
DATA_PREFIX="dx12p5"

# Independent chain seeds
SEED_BASELINE=42
SEED_MONITOR1=43
SEED_MONITOR2=44

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "$MODELS/baseline.pth" ]]; then
  echo "missing $MODELS/baseline.pth; run build_dx12p5.sh first" >&2
  exit 1
fi

case "$SURVEY" in
  baseline) SEED="$SEED_BASELINE" ;;
  monitor1) SEED="$SEED_MONITOR1" ;;
  monitor2) SEED="$SEED_MONITOR2" ;;
  *) echo "unknown survey: $SURVEY" >&2; exit 2 ;;
esac

SIMULT_FLAG=1
[[ "$SOURCE_MODE" == "sequential" ]] && SIMULT_FLAG=0
SHORT_FLAG=()
[[ "$SHORT_OUTPUT" == "1" ]] && SHORT_FLAG+=(--short_output)

SIGMA_JSON="$MODELS/sigma_${SURVEY}.json"
if [[ ! -f "$SIGMA_JSON" ]]; then
  "$PYTHON" -m dp4inversio4d calibrate --truth "$TRUTH" --survey "$SURVEY" \
    --source-mode "$SOURCE_MODE" --nsrc "$NSRC" --nrec "$NREC" \
    --nsimsrc "$NSIMSRC" --freq "$FREQ" --target-snr-db "$TARGET_SNR_DB" \
    --noise-type "$NOISE_TYPE" --nbl "$NBL" --tmax-ms "$TMAX_MS" \
    --deth-rec "$DETH_REC" --deth-src "$DETH_SRC" \
    --edge-margin-m "$EDGE_MARGIN_M" --water-depth-m "$WATER_DEPTH_M" \
    --checkpointing "$DATA_CHECKPOINTING" --num-checkpoints "$NUM_CHECKPOINTS"
fi
SIGMA="$("$PYTHON" -c "import json;print(repr(json.load(open('$SIGMA_JSON'))['sigma']))")"

attempt=0
until "$PYTHON" scripts/inversion/run_imaging.py --phase sample --cuda "$CUDA" \
  --experiment "${EXPERIMENT_PREFIX}_${SURVEY}" \
  --file_figs "${FILE_FIGS_PREFIX}_${SURVEY}" \
  --file_simut_data "${DATA_PREFIX}_${SURVEY}_simult.h5" \
  --file_seq_data "${DATA_PREFIX}_${SURVEY}_sequential.h5" \
  --vel_model marmousi --model_path "$MODELS/$SURVEY.pth" \
  --background_path "$MODELS/background.pth" \
  --noise_type "$NOISE_TYPE" --noise_seed "$SEED" --sigma "$SIGMA" --seed "$SEED" \
  --nbl "$NBL" --checkpointing "$CHECKPOINTING" \
  --num_checkpoints "$NUM_CHECKPOINTS" --tmax_ms "$TMAX_MS" --deth_rec "$DETH_REC" \
  --deth_src "$DETH_SRC" --edge_margin_m "$EDGE_MARGIN_M" \
  --np_kernel "$NP_KERNEL" --water_depth_m "$WATER_DEPTH_M" \
  --bg_sigma "$BG_SIGMA" \
  --max_itr "$MAX_ITR" --lr "$LR" --lr_final "$LR_FINAL" --wd "$WD" \
  --gamma "$GAMMA" --scale "$SCALE" --nsrc "$NSRC" --nrec "$NREC" \
  --nsimsrc "$NSIMSRC" --freq "$FREQ" --simult "$SIMULT_FLAG" \
  --run_all "$RUN_ALL" --retrain "$RETRAIN" --bg_linear "$BG_LINEAR" \
  "${SHORT_FLAG[@]}" \
  --upsample_scale "$UPSCALE" --coarse_output_scale "$COARSE_OUTPUT_SCALE" \
  --ch_encoding $CH_ENCODING --ch_decoding $CH_DECODING --ch_skip $CH_SKIP; do
  status=$?
  attempt=$((attempt+1))
  if (( attempt >= MAX_RETRIES )); then
    echo "run failed after $attempt attempts (status $status)" >&2
    exit "$status"
  fi
  echo "run failed (status $status); resuming from checkpoint in ${RETRY_WAIT_S}s" >&2
  sleep "$RETRY_WAIT_S"
done
