#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" scripts/inversion/run_imaging.py --phase sample --cuda 1 \
  --file_simut_data simult_src_31_rec_101_4Layers.h5 \
  --file_seq_data observed_seq_src_31_rec_101_4Layers.h5 \
  --file_figs 4L_maxvel_3.60 --lr 1e-3 --lr_final 1e-4 --wd 30 --sigma 0.1 \
  --max_itr 10000 --seed 42 --nsrc 31 --nrec 101 --vel_model 4layers \
  --retrain 1 --run_all 0 --simult 0 --scale 10.0 --upsample_scale 1 \
  --ch_encoding 8 8 8 --ch_decoding 8 8 8 --ch_skip 0 0 8 \
  --time_seismic 1600 --freq 11.0 --nsimsrc 31
