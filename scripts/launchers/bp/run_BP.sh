#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

MAX_ITR="${MAX_ITR:-50001}"
"$PYTHON_BIN" scripts/inversion/run_imaging.py --phase sample --cuda 1 --simult 0 \
  --file_simut_data simult_src_124_nrec_249_BP_3Hz.h5 \
  --file_seq_data observed_seq_src_124_nrec_249_BP_3Hz.h5 \
  --file_figs BP_seed_42 --lr 1e-3 --lr_final 1e-4 --wd 40 --sigma 1.5 \
  --max_itr "$MAX_ITR" --seed 42 --nsrc 124 --nrec 249 --vel_model bp \
  --retrain 1 --upsample_scale 1 --run_all 0 --scale 10 \
  --ch_encoding 16 16 --ch_decoding 16 16 --ch_skip 0 0 \
  --freq 3.0 --nsimsrc 124 --bg_linear 1 --dv 0.0 --coarse_output_scale 4
