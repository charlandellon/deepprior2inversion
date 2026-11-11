#!/usr/bin/env bash
# Shared launcher for one independent 4D chain.
# All tunable parameters live here, in VAR=value form, for easy editing.
set -euo pipefail

SURVEY="${1:?usage: run_survey.sh baseline|monitor1|monitor2}"

# Paths
PYTHON="python"
TRUTH="outputs/4d/marmousi4d_expanded_oil_water_white_E2"
MODELS="$TRUTH/truth/models"

# Acquisition
SOURCE_MODE="simult"          # sequential | simult
NSRC=100
NREC=400
NSIMSRC=100
FREQ=5.0
RUN_ALL=0                     # reuse the already generated observations; set 1 to regenerate
RETRAIN=0                     # never delete existing checkpoints/samples on relaunch
BG_LINEAR=1
TARGET_SNR_DB=6.0
SHORT_OUTPUT=1                # compact 4d/<experiment> paths for Windows MAX_PATH
MAX_RETRIES=5                 # automatic resumes after transient filesystem errors
RETRY_WAIT_S=30

# Inversion
MAX_ITR=25000
CUDA=1
LR=0.001
LR_FINAL=0.0001
WD=30.0
GAMMA=-0.3333
SCALE=10.0
UPSCALE=1
COARSE_OUTPUT_SCALE=1
CH_ENCODING="16 16 16"
CH_DECODING="16 16 16"
CH_SKIP="16 16 16"

# Experiment naming
EXPERIMENT_PREFIX="expanded4d"
FILE_FIGS_PREFIX="4d/expanded4d"
DATA_PREFIX="expanded4d"

# Independent chain seeds (also used for the white-noise realization)
SEED_BASELINE=42
SEED_MONITOR1=43
SEED_MONITOR2=44

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

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
  --background_path "$MODELS/background.pth" \
  --noise_type white_gaussian --sigma "$SIGMA" --seed "$SEED" \
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
