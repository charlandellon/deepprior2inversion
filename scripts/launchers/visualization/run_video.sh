#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

INPUT_DIR="${1:-data/images}"
OUTPUT_FILE="${2:-video_BP_inversion}"
"$PYTHON_BIN" scripts/visualization/video_inversion.py --dir_data_in "$INPUT_DIR" \
  --file_data_out "$OUTPUT_FILE" --duration 1.0
