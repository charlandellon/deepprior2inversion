#!/usr/bin/env bash
# Wait for deterministic FWI to finish, then resume the 800-receiver Deep Prior.
set -euo pipefail

FWI_PID="${1:?usage: resume_dp_after_fwi.sh <fwi_pid>}"
MODEL_FINAL="outputs/4d/deterministic_fwi/dx12p5_baseline/model_final.pth"
DP_LAUNCHER="dp4inversio4d/launchers/run_dx12p5.sh"
DP_LOG="data/checkpoints/4d/dx12p5_baseline_800rec_c32_resume.log"
SKIP_LOG="data/checkpoints/4d/dp_resume_skipped.log"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

while kill -0 "$FWI_PID" 2>/dev/null; do
  sleep 60
done

if [[ -f "$MODEL_FINAL" ]]; then
  bash "$DP_LAUNCHER" baseline > "$DP_LOG" 2>&1
else
  echo "FWI did not produce $MODEL_FINAL; Deep Prior not resumed." > "$SKIP_LOG"
fi
