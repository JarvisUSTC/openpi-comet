#!/usr/bin/env bash
# 单机 8 卡 PyTorch finetune 启动脚本
# 使用 torchrun + scripts/train_pytorch.py
cd /root/Training/memory0.1
source .venv/bin/activate 2>/dev/null || true

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_NAME="${1:-pi05_b1k-all_skills_mem_K6}"
EXP_NAME="${2:-openpi_mem_$(date +%Y%m%d_%H%M%S)}"

echo "Config: ${CONFIG_NAME}, exp_name: ${EXP_NAME}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Num GPUs: 8"

torchrun --nproc_per_node=8 scripts/train_pytorch.py "${CONFIG_NAME}" --exp_name="${EXP_NAME}"
