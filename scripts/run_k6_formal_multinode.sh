#!/usr/bin/env bash
set -euo pipefail

# K=6 formal training with multi-node support.
# Uses train_dist.py for JAX distributed training.
#
# Required env vars (injected by platform or set manually):
#   master_addr / MASTER_ADDR
#   master_port / MASTER_PORT
#   nnodes / WORLD_SIZE
#   node_rank / WORLD_RANK
#   nproc_per_node / NPROC_PER_NODE

CONFIG_NAME="${CONFIG_NAME:-pi05_b1k-k6-smoke-step50000}"
EXP_NAME="${EXP_NAME:-}"
OVERWRITE="${OVERWRITE:-true}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

# Map platform env vars
export MASTER_ADDR="${master_addr:-${MASTER_ADDR:-}}"
export MASTER_PORT="${master_port:-${MASTER_PORT:-}}"
export WORLD_SIZE="${nnodes:-${WORLD_SIZE:-}}"
export WORLD_RANK="${node_rank:-${WORLD_RANK:-${RANK:-}}}"
NPROC="${nproc_per_node:-${NPROC_PER_NODE:-8}}"

if [[ -z "${MASTER_ADDR}" || -z "${MASTER_PORT}" || -z "${WORLD_SIZE}" || -z "${WORLD_RANK}" ]]; then
  echo "[ERROR] Missing distributed env vars (MASTER_ADDR, MASTER_PORT, WORLD_SIZE, WORLD_RANK)" >&2
  exit 1
fi

# Set CUDA_VISIBLE_DEVICES if not already set
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  devs=""
  for ((i = 0; i < NPROC; i++)); do
    [[ -z "${devs}" ]] && devs="${i}" || devs="${devs},${i}"
  done
  export CUDA_VISIBLE_DEVICES="${devs}"
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export PYTHONUNBUFFERED=1

# Generate exp_name on rank 0, placeholder on others (train_dist.py broadcasts from rank 0)
if [[ -z "${EXP_NAME}" ]]; then
  if [[ "${WORLD_RANK}" == "0" ]]; then
    EXP_NAME="k6_formal_$(date +%Y%m%d_%H%M%S)"
  else
    EXP_NAME="placeholder"
  fi
fi

echo "[INFO] Config: ${CONFIG_NAME}"
echo "[INFO] Exp name: ${EXP_NAME}"
echo "[INFO] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[INFO] WORLD_SIZE=${WORLD_SIZE} WORLD_RANK=${WORLD_RANK}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

EXTRA_ARGS=()
if [[ "${OVERWRITE}" == "true" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi
if [[ "${WANDB_ENABLED}" == "true" ]]; then
  EXTRA_ARGS+=(--wandb-enabled)
else
  EXTRA_ARGS+=(--no-wandb-enabled)
fi

python -u scripts/train_dist.py \
  "${CONFIG_NAME}" \
  --exp_name "${EXP_NAME}" \
  "${EXTRA_ARGS[@]}"
