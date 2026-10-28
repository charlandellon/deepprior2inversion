#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${ENV_NAME:-deepprior2inversion}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RECREATE=0
UPDATE_ENV=0

usage() {
  cat <<'EOF'
Uso: bash setup_env.sh [--update|--recreate]

  (sem opções)  cria o ambiente se não existir; reaproveita se já existir
  --update      atualiza o ambiente existente
  --recreate    remove e recria o ambiente
  --help        mostra esta mensagem

Variáveis: ENV_NAME, PYTHON_BIN, CONDA_BASE
EOF
}

for arg in "$@"; do
  case "$arg" in
    --update) UPDATE_ENV=1 ;;
    --recreate) RECREATE=1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Opção inválida: $arg" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$RECREATE" == "1" && "$UPDATE_ENV" == "1" ]]; then
  echo "Use apenas --update ou --recreate." >&2
  exit 1
fi

if [[ ! -f environment.yml ]]; then
  echo "environment.yml não encontrado em $ROOT_DIR" >&2
  exit 1
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Aviso: este script foi feito para WSL/Linux; o environment.yml usa cudatoolkit Linux." >&2
fi

if ! command -v conda >/dev/null 2>&1; then
  for prefix in "${CONDA_BASE:-}" "$HOME/anaconda3" "$HOME/miniconda3" "/opt/conda"; do
    if [[ -n "$prefix" && -f "$prefix/etc/profile.d/conda.sh" ]]; then
      source "$prefix/etc/profile.d/conda.sh"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda não encontrado. Ative o Conda ou defina CONDA_BASE." >&2
  exit 1
fi

env_exists() {
  conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx "$ENV_NAME"
}

if env_exists; then
  if [[ "$RECREATE" == "1" ]]; then
    echo "Removendo ambiente $ENV_NAME..."
    conda env remove -n "$ENV_NAME" -y
    conda env create -n "$ENV_NAME" -f environment.yml
  elif [[ "$UPDATE_ENV" == "1" ]]; then
    echo "Atualizando ambiente $ENV_NAME..."
    conda env update -n "$ENV_NAME" -f environment.yml --prune
  else
    echo "Ambiente $ENV_NAME já existe; reaproveitando."
  fi
else
  echo "Criando ambiente $ENV_NAME..."
  conda env create -n "$ENV_NAME" -f environment.yml
fi

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u

"$PYTHON_BIN" -m pip install -e "$ROOT_DIR"

"$PYTHON_BIN" - <<'PY'
import devito
import torch

import dp4inversion

print(f'dp4inversion: {dp4inversion.__file__}')
print(f'devito: {devito.__version__}')
print(f'torch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
PY

cat <<EOF
Ambiente pronto: $ENV_NAME
Ative com: conda activate $ENV_NAME
Próximo passo: CUDA=1 bash run.sh bp
EOF
