#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/vepfs_uv_env.sh"
cd "${REPO_ROOT}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

CONFIG_NAME="${1:-pi05_b1k-knowledge_insulation-vqa-joint}"
EXP_NAME="${2:-${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)}"

resolve_first_set() {
  for value in "$@"; do
    if [[ -n "${value}" ]]; then
      printf '%s\n' "${value}"
      return 0
    fi
  done
  return 1
}

MASTER_ADDR="$(
  resolve_first_set \
    "${MASTER_ADDR:-}" \
    "${COORDINATOR_ADDRESS:-}" \
    "${SERVICE_PREFIX:-}${SERVICE_PREFIX:+-0.}${SUBDOMAIN:-}"
)"

WORLD_SIZE="$(
  resolve_first_set \
    "${WORLD_SIZE:-}" \
    "${NNODES:-}" \
    "${LEPTON_JOB_TOTAL_WORKERS:-}" \
    "${SLURM_NNODES:-}"
)"

WORLD_RANK="$(
  resolve_first_set \
    "${WORLD_RANK:-}" \
    "${NODE_RANK:-}" \
    "${RANK:-}" \
    "${LEPTON_JOB_WORKER_INDEX:-}" \
    "${SLURM_NODEID:-}" \
    "0"
)"

MASTER_PORT="${MASTER_PORT:-12350}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
OPENPI_FAST_TOKENIZER_PATH="${OPENPI_FAST_TOKENIZER_PATH:-/vepfs-C/Jiawei/fast}"
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
PYTHONPATH="${PYTHONPATH:-src:packages/openpi-client/src}"

if [[ -z "${MASTER_ADDR}" ]]; then
  echo "MASTER_ADDR 未设置。" >&2
  echo "请显式导出 MASTER_ADDR，或在 Lepton 环境下提供 SERVICE_PREFIX/SUBDOMAIN。" >&2
  exit 1
fi

if [[ -z "${WORLD_SIZE}" ]]; then
  echo "WORLD_SIZE 未设置。" >&2
  echo "请显式导出 WORLD_SIZE/NNODES，或在调度环境中提供 LEPTON_JOB_TOTAL_WORKERS/SLURM_NNODES。" >&2
  exit 1
fi

export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE
export WORLD_RANK
export CUDA_VISIBLE_DEVICES
export OPENPI_FAST_TOKENIZER_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE
export XLA_PYTHON_CLIENT_ALLOCATOR
export PYTHONPATH

echo "Repo root: ${REPO_ROOT}"
echo "Config: ${CONFIG_NAME}"
echo "Exp name: ${EXP_NAME}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "WORLD_RANK: ${WORLD_RANK}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "OPENPI_FAST_TOKENIZER_PATH: ${OPENPI_FAST_TOKENIZER_PATH}"
echo "XLA_PYTHON_CLIENT_PREALLOCATE: ${XLA_PYTHON_CLIENT_PREALLOCATE}"
echo "XLA_PYTHON_CLIENT_ALLOCATOR: ${XLA_PYTHON_CLIENT_ALLOCATOR}"

# 说明：
# - 这个脚本假设“每台机器只启动一个 Python 进程”，该进程使用本机所有可见 GPU。
# - 因此 WORLD_SIZE 应该等于机器数（节点数），而不是总 GPU 数。
# - fsdp_devices / batch_size 仍然通过 config.py 或命令行 override 控制。

python scripts/train_dist.py "${CONFIG_NAME}" --exp_name="${EXP_NAME}" --overwrite
