#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${1:-bp}"

case "$MODEL" in
  bp)
    FREQ="${FREQ:-3.0}"
    NSRC="${NSRC:-117}"
    NREC="${NREC:-234}"
    NSIMSRC="${NSIMSRC:-$NSRC}"
    SIGMA="${SIGMA:-1.5}"
    SIMULT="${SIMULT:-0}"
    BG_LINEAR="${BG_LINEAR:-1}"
    COARSE_OUTPUT_SCALE="${COARSE_OUTPUT_SCALE:-4}"
    FILE_FIGS="${FILE_FIGS:-BP_seed_42}"
    SIM_FILE="${SIM_FILE:-simult_src_117_nrec_234_BP_3Hz.h5}"
    SEQ_FILE="${SEQ_FILE:-observed_seq_src_117_nrec_234_BP_3Hz.h5}"
    MODEL_SCRIPT="scripts/velocity_models/gerando_BP.py"
    MODEL_FILES=(
      "data/velocity_model/BP_model_freq_3_v1.pth"
      "data/velocity_model/BP-smooth-linear_freq_3_v1.pth"
    )
    ;;
  marmousi)
    FREQ="${FREQ:-5.0}"
    NSRC="${NSRC:-100}"
    NREC="${NREC:-400}"
    NSIMSRC="${NSIMSRC:-$NSRC}"
    SIGMA="${SIGMA:-0.5}"
    SIMULT="${SIMULT:-1}"
    BG_LINEAR="${BG_LINEAR:-2}"
    COARSE_OUTPUT_SCALE="${COARSE_OUTPUT_SCALE:-1}"
    FILE_FIGS="${FILE_FIGS:-marmousi_seed_42_bg_superlinear}"
    SIM_FILE="${SIM_FILE:-simult_src_100_rec_400_marmousi_5Hz.h5}"
    SEQ_FILE="${SEQ_FILE:-observed_seq_src_100_rec_400_marmousi_5Hz.h5}"
    MODEL_SCRIPT="scripts/velocity_models/gerando_marmousi.py"
    MODEL_FILES=(
      "data/velocity_model/marmousi2-model-true-large_freq_5.pth"
      "data/velocity_model/marmousi2-model-smooth-linear_freq_5.pth"
    )
    ;;
  *)
    echo "Uso: bash run.sh [bp|marmousi]" >&2
    exit 1
    ;;
esac

MAX_ITR="${MAX_ITR:-50001}"
LR="${LR:-1e-3}"
LR_FINAL="${LR_FINAL:-1e-4}"
WD="${WD:-30}"
SEED="${SEED:-42}"
SCALE="${SCALE:-10}"
UPSAMPLE_SCALE="${UPSAMPLE_SCALE:-1}"
RETRAIN="${RETRAIN:-1}"
RUN_ALL="${RUN_ALL:-1}"
RUN_STATS="${RUN_STATS:-1}"
GENERATE_MODEL="${GENERATE_MODEL:-1}"

read -r -a CH_ENCODING_ARR <<< "${CH_ENCODING:-16 16 16}"
read -r -a CH_DECODING_ARR <<< "${CH_DECODING:-16 16 16}"
read -r -a CH_SKIP_ARR <<< "${CH_SKIP:-0 0 16}"

if ! "$PYTHON_BIN" -c "import dp4inversion" >/dev/null 2>&1; then
  echo "Pacote dp4inversion não encontrado. Ative o ambiente e rode: pip install -e \"$ROOT_DIR\"" >&2
  exit 1
fi

if [[ "$GENERATE_MODEL" == "1" ]]; then
  if ! command -v wget >/dev/null 2>&1; then
    echo "wget é necessário para baixar os modelos de velocidade." >&2
    exit 1
  fi
  echo "[1/3] Baixando e gerando o modelo de velocidade: $MODEL"
  "$PYTHON_BIN" "$MODEL_SCRIPT"
fi

for file in "${MODEL_FILES[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "Arquivo de modelo ausente: $file" >&2
    exit 1
  fi
done

COMMON_ARGS=(
  --phase sample
  --cuda "${CUDA:-1}"
  --file_simut_data "$SIM_FILE"
  --file_seq_data "$SEQ_FILE"
  --file_figs "$FILE_FIGS"
  --lr "$LR"
  --lr_final "$LR_FINAL"
  --wd "$WD"
  --sigma "$SIGMA"
  --max_itr "$MAX_ITR"
  --seed "$SEED"
  --nsrc "$NSRC"
  --nrec "$NREC"
  --vel_model "$MODEL"
  --retrain "$RETRAIN"
  --upsample_scale "$UPSAMPLE_SCALE"
  --run_all "$RUN_ALL"
  --scale "$SCALE"
  --ch_encoding "${CH_ENCODING_ARR[@]}"
  --ch_decoding "${CH_DECODING_ARR[@]}"
  --ch_skip "${CH_SKIP_ARR[@]}"
  --freq "$FREQ"
  --simult "$SIMULT"
  --nsimsrc "$NSIMSRC"
  --bg_linear "$BG_LINEAR"
  --coarse_output_scale "$COARSE_OUTPUT_SCALE"
)

echo "[2/3] Gerando dados sísmicos e executando a inversão: $MODEL"
"$PYTHON_BIN" scripts/inversion/run_imaging.py "${COMMON_ARGS[@]}"

if [[ "$RUN_STATS" == "1" ]]; then
  STAT_ARGS=(
    --file_figs "$FILE_FIGS"
    --lr "$LR"
    --lr_final "$LR_FINAL"
    --wd "$WD"
    --sigma "$SIGMA"
    --max_itr "$MAX_ITR"
    --seed "$SEED"
    --nsrc "$NSRC"
    --nrec "$NREC"
    --vel_model "$MODEL"
    --scale "$SCALE"
    --upsample_scale "$UPSAMPLE_SCALE"
    --ch_encoding "${CH_ENCODING_ARR[@]}"
    --ch_decoding "${CH_DECODING_ARR[@]}"
    --ch_skip "${CH_SKIP_ARR[@]}"
    --freq "$FREQ"
    --simult "$SIMULT"
  )
  echo "[3/3] Calculando os índices estatísticos: $MODEL"
  "$PYTHON_BIN" scripts/analysis/statistical_indice.py "${STAT_ARGS[@]}"
fi

echo "Pipeline concluído: $MODEL"
