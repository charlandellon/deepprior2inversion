#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" scripts/modeling/plot_seismic_source.py --phase sample --cuda 1 \
  --file_simut_data simult_src_31_rec_101_4Layers_10Hz.h5 \
  --file_seq_data observed_seq_src_31_rec_101_4Layers_10Hz.h5 \
  --file_figs 4Layers_velocidade --lr 1e-3 --lr_final 1e-4 --wd 30 --sigma 0.1 \
  --max_itr 10000 --seed 42 --nsrc 31 --nrec 101 --vel_model 4layers \
  --retrain 1 --run_all 1 --simult 1 --scale 5.0 --upsample_scale 1 \
  --ch_encoding 16 32 64 --ch_decoding 16 32 64 --ch_skip 0 0 64 \
  --time_seismic 950.0 --nsimsrc 31 --idx_source 15
