#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" scripts/analysis/plotar_rmse_4layer.py \
  --file_figs 4L_maxvel_3.60_PRelu_nomask_nbl_30_ --lr 1e-3 --lr_final 1e-4 --wd 30 --sigma 0.1 \
  --max_itr 10000 --seed 42 --nsrc 31 --nrec 50 --vel_model 4layers --scale 5.0 \
  --upsample_scale 1 --ch_encoding 16 32 64 --ch_decoding 16 32 64 \
  --ch_skip 0 0 64 --time_seismic 1600 --freq 11.0
