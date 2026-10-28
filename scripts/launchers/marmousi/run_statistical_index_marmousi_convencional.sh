#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" scripts/analysis/statistical_indice_convencional.py \
  --file_figs marmousi_seed_42_bg_linear --lr 1e-3 \
  --lr_final 1e-4 --wd 30 --sigma 0.5 --max_itr 50001 --seed 42 --nsrc 100 \
  --nrec 400 --vel_model marmousi --scale 10 --freq 5.0 --nsimsrc 100 --bg_linear 1
