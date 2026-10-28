#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" scripts/analysis/statistical_indice.py \
  --file_figs tmp_2_marmousi_src_200_rec_400_sigma_2_seed_42 \
  --lr 1e-3 --lr_final 1e-4 --wd 30 --sigma 2.0 \
  --max_itr 20000 --seed 42 --nsrc 200 --nrec 400 --vel_model marmousi --freq 5.0
