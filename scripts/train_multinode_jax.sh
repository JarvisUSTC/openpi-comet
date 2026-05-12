#!/usr/bin/env bash
# Aladdin/JAX multi-node entrypoint.
# One process is launched per replica and uses all GPUs visible on that replica.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

CONFIG_NAME="${1:?missing config name}"
EXP_NAME="${2:?missing exp name}"

ALADDIN_MASTER_ADDR="${master_addr:-${MASTER_ADDR:-}}"
ALADDIN_MASTER_PORT="${master_port:-${MASTER_PORT:-}}"
ALADDIN_NNODES="${GROUP_POD_SIZE:-${nnodes:-${WORLD_SIZE:-}}}"
ALADDIN_NODE_RANK="${GROUP_POD_INDEX:-${node_rank:-${WORLD_RANK:-${RANK:-}}}}"
ALADDIN_NPROC_PER_NODE="${nproc_per_node:-${NPROC_PER_NODE:-${MLP_WORKER_GPU:-}}}"

if [[ -z "${ALADDIN_MASTER_ADDR}" ]]; then
  echo "ERROR: master_addr or MASTER_ADDR is required." >&2
  exit 2
fi
if [[ -z "${ALADDIN_MASTER_PORT}" ]]; then
  echo "ERROR: master_port or MASTER_PORT is required." >&2
  exit 2
fi
if [[ -z "${ALADDIN_NNODES}" ]]; then
  echo "ERROR: GROUP_POD_SIZE, nnodes, or WORLD_SIZE is required." >&2
  exit 2
fi
if [[ -z "${ALADDIN_NODE_RANK}" ]]; then
  echo "ERROR: GROUP_POD_INDEX, node_rank, WORLD_RANK, or RANK is required." >&2
  exit 2
fi

export MASTER_ADDR="${ALADDIN_MASTER_ADDR}"
export MASTER_PORT="${ALADDIN_MASTER_PORT}"
export WORLD_SIZE="${ALADDIN_NNODES}"
export WORLD_RANK="${ALADDIN_NODE_RANK}"

make_cvd() {
  local n="${1:-8}"
  local cvd="0"
  local i
  for ((i = 1; i < n; i++)); do
    cvd+=",${i}"
  done
  printf '%s\n' "${cvd}"
}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$(make_cvd "${ALADDIN_NPROC_PER_NODE:-8}")"
fi

export PYTHONPATH="${PYTHONPATH:-src:packages/openpi-client/src}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/root/.cache/openpi}"
export OPENPI_FAST_TOKENIZER_PATH="${OPENPI_FAST_TOKENIZER_PATH:-/root/Models/pi_fast_tokenizer}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export JAX_TRACEBACK_FILTERING="${JAX_TRACEBACK_FILTERING:-off}"
export WANDB_ENABLED="${WANDB_ENABLED:-true}"

if [[ "${WANDB_ENABLED}" != "false" && "${WANDB_ENABLED}" != "0" ]]; then
  export WANDB_MODE="${WANDB_MODE:-online}"
  : "${WANDB_API_KEY:?WANDB_API_KEY is required when WANDB_ENABLED=true.}"
fi

TRAIN_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--overwrite)
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--resume)
fi
if [[ "${WANDB_ENABLED}" == "false" || "${WANDB_ENABLED}" == "0" ]]; then
  TRAIN_ARGS+=(--no-wandb-enabled)
fi

echo "CONFIG_NAME=${CONFIG_NAME}"
echo "EXP_NAME=${EXP_NAME}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "WORLD_SIZE=${WORLD_SIZE}"
echo "WORLD_RANK=${WORLD_RANK}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "OPENPI_FAST_TOKENIZER_PATH=${OPENPI_FAST_TOKENIZER_PATH}"
echo "WANDB_ENABLED=${WANDB_ENABLED}"

python -u scripts/train_dist.py "${CONFIG_NAME}" --exp_name="${EXP_NAME}" "${TRAIN_ARGS[@]}"
