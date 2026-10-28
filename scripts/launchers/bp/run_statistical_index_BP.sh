#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" scripts/analysis/statistical_indice.py \
  --file_figs BP_seed_42 --lr 1e-3 --lr_final 1e-4 --wd 40 --sigma 1.5 \
  --max_itr 50001 --seed 42 --nsrc 124 --nrec 249 --vel_model bp \
  --upsample_scale 1 --run_all 0 --scale 20 \
  --ch_encoding 16 16 16 --ch_decoding 16 16 16 --ch_skip 0 0 0 --freq 3.0
