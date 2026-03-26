#!/usr/bin/env bash
# 单机 8 卡 OpenPi finetune 启动脚本
# 使用 scripts/train.py：单进程、JAX 自动使用全部可见 GPU（数据+可选 FSDP 分片）
#
set -euo pipefail

# 注意（config 里需满足）：
#   - batch_size 必须能被 GPU 数整除，例如 8 卡时 batch_size=8*32=256
#   - fsdp_devices：设为 8 时模型 8 路 FSDP 分片（省显存）；设为 1 时仅数据并行

# ---------- 环境变量（8 卡）---------
# 指定使用的 GPU，8 卡时通常为 0-7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 限制 JAX 占用的 GPU 显存比例，避免 OOM；按需调整（0.9 即 90%）
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# 可选：多卡时建议关闭预分配，让 JAX 按需分配（data_loader 里 worker 会设 PREALLOCATE=false，主进程也可设）
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# 可选：JAX 编译缓存目录，多卡/多任务时可减少重复编译
# export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/jax}"

# ---------- 运行目录 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source .venv/bin/activate

# ---------- 配置与实验名 ----------
# 对应 config.py 里注册的 TrainConfig 的 name（例如 pi05_b1k-turning_on_radio）
CONFIG_NAME="${1:-pi05_b1k-turning_on_radio}"
# 实验名，用于 checkpoint 目录等
EXP_NAME="${2:-openpi_$(date +%Y%m%d_%H%M%S)}"

echo "Config: ${CONFIG_NAME}, exp_name: ${EXP_NAME}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "XLA_PYTHON_CLIENT_MEM_FRACTION: ${XLA_PYTHON_CLIENT_MEM_FRACTION}"

# ---------- 启动训练 ----------
# 使用 uv 运行（与 README 一致）；若已 source .venv/bin/activate 可改为: python scripts/train.py ...
# uv run scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}" # 运行一次即可
uv run scripts/train.py "${CONFIG_NAME}" --exp_name="${EXP_NAME}"

# 可选：从 checkpoint 恢复
# uv run scripts/train.py "${CONFIG_NAME}" --exp_name="${EXP_NAME}" --resume
