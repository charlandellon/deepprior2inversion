#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" scripts/modeling/plot_seismic_source.py --phase sample --cuda 1 \
  --file_figs BP_seed_42 --lr 1e-3 --lr_final 1e-4 --wd 30 \
  --max_itr 50001 --seed 42 --nsrc 117 --nrec 234 --vel_model bp \
  --retrain 1 --upsample_scale 1 --run_all 0 --scale 10 \
  --ch_encoding 16 16 16 --ch_decoding 16 16 16 --ch_skip 0 0 16 \
  --freq 3.0 --bg_linear 1 --idx_source 58 --dv 0 --tmax 10 --snr 2.4
