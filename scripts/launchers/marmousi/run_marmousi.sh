#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

MAX_ITR="${MAX_ITR:-50001}"
"$PYTHON_BIN" scripts/inversion/run_imaging.py --phase sample --cuda 1 \
  --file_simut_data simult_src_100_rec_400_marmousi_5Hz.h5 \
  --file_seq_data observed_seq_src_100_rec_400_marmousi_5Hz.h5 \
  --file_figs marmousi_seed_42_bg_superlinear --lr 1e-3 \
  --lr_final 1e-4 --wd 30 --sigma 0.5 --max_itr "$MAX_ITR" --seed 42 --nsrc 100 \
  --nrec 400 --vel_model marmousi --retrain 1 --upsample_scale 1 --run_all 0 \
  --scale 10 --ch_encoding 16 16 16 --ch_decoding 16 16 16 --ch_skip 0 0 16 \
  --freq 5.0 --nsimsrc 100 --bg_linear 2
