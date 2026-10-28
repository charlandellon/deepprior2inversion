#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" scripts/inversion/run_imaging_convencional.py --phase sample --cuda 1 --simult 1 \
  --file_simut_data simult_src_100_rec_400_marmousi_5Hz.h5 \
  --file_seq_data observed_seq_src_100_rec_400_marmousi_5Hz.h5 \
  --file_figs marmousi_seed_42_bg_superlinear --lr 1e-3 \
  --lr_final 1e-4 --wd 30 --sigma 1.0 --max_itr 50001 --seed 42 --nsrc 100 \
  --nrec 400 --vel_model marmousi --retrain 1 --run_all 0 \
  --scale 10 --freq 5.0 --nsimsrc 100 --bg_linear 2
