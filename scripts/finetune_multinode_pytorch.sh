#!/usr/bin/env bash
# 多节点多卡 PyTorch finetune 启动脚本
# 适配 Aladdin 平台 Run Task 模式，平台自动注入:
#   $nproc_per_node, $nnodes, $node_rank, $master_addr, $master_port
cd /root/Training/memory0.1
source .venv/bin/activate 2>/dev/null || true

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# WandB API key: set WANDB_API_KEY env var before running, or use `wandb login`
# export WANDB_API_KEY="your_key_here"

# Patch transformers with MEM-modified SigLIP/PaliGemma on every node
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
      .venv/lib/python3.11/site-packages/transformers/
echo "transformers_replace patched on node $(hostname)"

CONFIG_NAME="${1:-pi05_b1k-all_skills_mem_K6}"
EXP_NAME="${2:-openpi_mem_$(date +%Y%m%d_%H%M%S)}"

NPROC=${nproc_per_node:-8}
NNODES=${nnodes:-1}
NODE_RANK=${node_rank:-0}
MASTER_ADDR=${master_addr:-localhost}
MASTER_PORT=${master_port:-29500}
TOTAL_GPUS=$((NPROC * NNODES))

echo "============================================"
echo "Config: ${CONFIG_NAME}"
echo "Exp name: ${EXP_NAME}"
echo "Nodes: ${NNODES}, GPUs per node: ${NPROC}, Total GPUs: ${TOTAL_GPUS}"
echo "Node rank: ${NODE_RANK}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "============================================"

torchrun \
  --nproc_per_node="${NPROC}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  scripts/train_pytorch.py "${CONFIG_NAME}" --exp_name="${EXP_NAME}"
