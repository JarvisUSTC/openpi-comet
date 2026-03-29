#!/usr/bin/env bash
# Aladdin task entrypoint: evaluate stop_prob stats on dataset batches.
#
# Usage (inside container):
#   bash scripts/aladdin/eval_stop_prob_dataset.sh --config-name ... --params-dir .../params

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export JAX_TRACEBACK_FILTERING="${JAX_TRACEBACK_FILTERING:-off}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/aladdin_logs}"
mkdir -p "${LOG_DIR}"
EVAL_NAME="${EVAL_NAME:-eval_stop_prob_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EVAL_NAME}.log}"

echo "Eval name: ${EVAL_NAME}"
echo "Log file:  ${LOG_FILE}"

set -o pipefail
uv run scripts/eval_stop_prob_dataset.py "$@" 2>&1 | tee -a "${LOG_FILE}"
