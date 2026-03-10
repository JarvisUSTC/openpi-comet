#!/usr/bin/env bash
# 单机 8 卡 OpenPi Knowledge Insulation (KI) finetune 启动脚本
# 使用 scripts/train.py：单进程、JAX 自动使用全部可见 GPU（数据+可选 FSDP 分片）
# 配置：pi0.5 + knowledge_insulation=True（backbone 用 FAST 离散 action tokens，action expert 梯度不回传）
#
# 注意（config 里需满足）：
#   - batch_size 必须能被 GPU 数整除，例如 8 卡时 batch_size=8*32=256
#   - fsdp_devices：设为 8 时模型 8 路 FSDP 分片（省显存）；设为 1 时仅数据并行
cd /root/Training/openpi-comet
source .venv/bin/activate

set -euo pipefail

# ---------- 环境变量（8 卡）---------
# 指定使用的 GPU，8 卡时通常为 0-7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 限制 JAX 占用的 GPU 显存比例，避免 OOM；按需调整（0.9 即 90%）
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# 无外网时：提前在有网机器执行 scripts/download_fast_tokenizer.sh /path/to/fast_tokenizer，
# 将目录拷到集群后设置下面环境变量，避免训练时连 HuggingFace 报错
export OPENPI_FAST_TOKENIZER_PATH="/root/Models/pi_fast_tokenizer"

# 可选：多卡时建议关闭预分配，让 JAX 按需分配（data_loader 里 worker 会设 PREALLOCATE=false，主进程也可设）
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# 可选：JAX 编译缓存目录，多卡/多任务时可减少重复编译
# export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/jax}"

# ---------- 运行目录 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---------- 配置与实验名 ----------
# Knowledge Insulation 配置（config.py 中 name=pi05_b1k-knowledge_insulation）
CONFIG_NAME="${1:-pi05_b1k-knowledge_insulation-all_skills}"
# 实验名，用于 checkpoint 目录等
EXP_NAME="${2:-openpi_knowledge_insulation-all_skills_$(date +%Y%m%d_%H%M%S)}"

echo "Config: ${CONFIG_NAME}, exp_name: ${EXP_NAME}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "XLA_PYTHON_CLIENT_MEM_FRACTION: ${XLA_PYTHON_CLIENT_MEM_FRACTION}"

# ---------- 启动训练 ----------
# 使用 uv 运行（与 README 一致）；若已 source .venv/bin/activate 可改为: python scripts/train.py ...
# uv run scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}" # 运行一次即可
uv run scripts/train.py "${CONFIG_NAME}" --exp_name="${EXP_NAME}"

# 可选：从 checkpoint 恢复
# uv run scripts/train.py "${CONFIG_NAME}" --exp_name="${EXP_NAME}" --resume
