#!/usr/bin/env bash
# Monitor-chain launcher for the shared-latent / baseline-BG experiment.
# Uses the same z and the same 2D velocity sample (drawn from the baseline
# post-burn-in posterior) for monitor1 and monitor2. The network weights are
# initialized from each chain seed; the BG logit map anchors the starting model.
set -euo pipefail

SURVEY="${1:?usage: run_survey_c32s0_g2_bg.sh monitor1|monitor2}"

# Paths
PYTHON="python"
TRUTH="outputs/4d/marmousi4d_expanded_oil_water_white_E2"
MODELS="$TRUTH/truth/models"
BG_PATH="$MODELS/bg_from_baseline.pth"
Z_PATH="$MODELS/z_baseline.pt"

# Acquisition
SOURCE_MODE="simult"          # sequential | simult
NSRC=100
NREC=400
NSIMSRC=100
FREQ=5.0
RUN_ALL=0                     # reuse the already generated observations; set 1 to regenerate
RETRAIN=0                     # never delete existing checkpoints/samples on relaunch
BG_LINEAR=0                   # keep the full 2D baseline sample as background
TARGET_SNR_DB=6.0
SHORT_OUTPUT=1                # compact 4d/<experiment> paths for Windows MAX_PATH
MAX_RETRIES=5                 # automatic resumes after transient filesystem errors
RETRY_WAIT_S=30

# Inversion
MAX_ITR=25000
CUDA=1
LR=0.001
LR_FINAL=0.0001
GAMMA=-2.0
LR_DECAY_END=17500
WD=30.0
SCALE=1.0
UPSCALE=1
COARSE_OUTPUT_SCALE=1
CH_ENCODING="32 32 32"
CH_DECODING="32 32 32"
CH_SKIP="0 0 32"

# Experiment naming (same tag as the baseline so stats can compare all three)
EXPERIMENT_PREFIX="expanded4d_s1_c32s0_g2"
FILE_FIGS_PREFIX="4d/expanded4d_s1_c32s0_g2"
DATA_PREFIX="expanded4d"

# Independent chain seeds (also used for the white-noise realization)
SEED_MONITOR1=43
SEED_MONITOR2=44

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

case "$SURVEY" in
  monitor1) SEED="$SEED_MONITOR1" ;;
  monitor2) SEED="$SEED_MONITOR2" ;;
  *) echo "this launcher only accepts monitor1 or monitor2" >&2; exit 2 ;;
esac

if [[ ! -f "$BG_PATH" ]]; then
  echo "missing baseline background: $BG_PATH" >&2
  exit 2
fi
if [[ ! -f "$Z_PATH" ]]; then
  echo "missing shared latent variable: $Z_PATH" >&2
  exit 2
fi

SIMULT_FLAG=1
[[ "$SOURCE_MODE" == "sequential" ]] && SIMULT_FLAG=0
SHORT_FLAG=()
[[ "$SHORT_OUTPUT" == "1" ]] && SHORT_FLAG+=(--short_output)

SIGMA_JSON="$MODELS/sigma_${SURVEY}.json"
if [[ ! -f "$SIGMA_JSON" ]]; then
  "$PYTHON" -m dp4inversio4d calibrate --truth "$TRUTH" --survey "$SURVEY" \
    --source-mode "$SOURCE_MODE" --nsrc "$NSRC" --nrec "$NREC" \
    --nsimsrc "$NSIMSRC" --freq "$FREQ" --target-snr-db "$TARGET_SNR_DB"
fi
SIGMA="$("$PYTHON" -c "import json;print(repr(json.load(open('$SIGMA_JSON'))['sigma']))")"

attempt=0
until "$PYTHON" scripts/inversion/run_imaging.py --phase sample --cuda "$CUDA" \
  --experiment "${EXPERIMENT_PREFIX}_${SURVEY}" \
  --file_figs "${FILE_FIGS_PREFIX}_${SURVEY}" \
  --file_simut_data "${DATA_PREFIX}_${SURVEY}_simult.h5" \
  --file_seq_data "${DATA_PREFIX}_${SURVEY}_sequential.h5" \
  --vel_model marmousi --model_path "$MODELS/$SURVEY.pth" \
  --background_path "$BG_PATH" --z_path "$Z_PATH" \
  --noise_type white_gaussian --sigma "$SIGMA" --seed "$SEED" \
  --max_itr "$MAX_ITR" --lr "$LR" --lr_final "$LR_FINAL" \
  --gamma "$GAMMA" --lr_decay_end "$LR_DECAY_END" --wd "$WD" \
  --scale "$SCALE" --nsrc "$NSRC" --nrec "$NREC" \
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
