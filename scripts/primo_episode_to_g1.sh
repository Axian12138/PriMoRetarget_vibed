#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH=src
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python_bin="${PRIMO_PYTHON:-python3}"
if [[ -z "${PRIMO_PYTHON:-}" && -x .venv/bin/python ]]; then
  python_bin=.venv/bin/python
fi
exec "$python_bin" -m primoretarget.episode "$@"
