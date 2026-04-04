#!/usr/bin/env bash
# Aladdin task entrypoint: multi-node JAX training for place_surface group WITH stop supervision + prompt aug.
#
# This script is meant to be used with:
#   aladdin task --replicas <N> ... -f scripts/aladdin/train_multinode_place_surface_stop_promptaug.sh
#
# It delegates to `scripts/train_multinode_jax.sh`, which expects Aladdin to inject:
#   nproc_per_node, nnodes, node_rank, master_addr, master_port

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

export CONFIG_NAME="${CONFIG_NAME:-pi05_b1k-sampled_skill_group-stop-full-pretrained-prompt-aug}"
export EXP_NAME="${EXP_NAME:-place_surface_stop_promptaug_multinode_$(date +%Y%m%d_%H%M%S)}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/aladdin_logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXP_NAME}.log}"

echo "Config:   ${CONFIG_NAME}"
echo "Exp name: ${EXP_NAME}"
echo "Log file: ${LOG_FILE}"

set -o pipefail
bash "${REPO_ROOT}/scripts/train_multinode_jax.sh" "${CONFIG_NAME}" "${EXP_NAME}" "$@" 2>&1 | tee -a "${LOG_FILE}"

