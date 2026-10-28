#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" scripts/analysis/plotar_rmse_marmousi_convencional_deep_prior.py \
  --file_figs marmousi_seed_42_bg_linear --lr 1e-3 \
  --lr_final 1e-4 --wd 30 --sigma 0.5 --max_itr 50001 --seed 42 --nsrc 100 \
  --nrec 400 --vel_model marmousi --upsample_scale 1 --freq 5.0 \
  --scale 10 --ch_encoding 16 32 64 --ch_decoding 16 32 64 --ch_skip 0 0 64
