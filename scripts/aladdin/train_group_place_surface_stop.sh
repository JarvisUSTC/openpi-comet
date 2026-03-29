#!/usr/bin/env bash
# Aladdin task entrypoint: train Pi05 skill-group (place_surface) WITH stop supervision.
#
# This script is meant to be used with:
#   aladdin task ... -f scripts/aladdin/train_group_place_surface_stop.sh
#
# It writes a local log file under `outputs/aladdin_logs/` so you can tail it from the host if `--work-dir`
# points to a shared filesystem path.

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

# Make Python output visible and tracebacks complete.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export JAX_TRACEBACK_FILTERING="${JAX_TRACEBACK_FILTERING:-off}"

# Avoid torch inductor spawning compile worker subprocesses in restricted environments.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

export CONFIG_NAME="${CONFIG_NAME:-pi05_b1k-sampled_skill_group-stop-full-pretrained}"
export EXP_NAME="${EXP_NAME:-pretrain_full_place_surface_stop_$(date +%Y%m%d_%H%M%S)}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/aladdin_logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXP_NAME}.log}"

echo "Config:   ${CONFIG_NAME}"
echo "Exp name: ${EXP_NAME}"
echo "Log file: ${LOG_FILE}"

set -o pipefail
bash "${REPO_ROOT}/scripts/full_groups/group_place_surface_stop.sh" "$@" 2>&1 | tee -a "${LOG_FILE}"

