#!/usr/bin/env bash
# 多机多卡（multi-node multi-GPU）JAX 训练启动脚本（默认：1 机 = 1 进程，进程内使用该机所有可见 GPU）
# 依赖 scripts/train_dist.py（JAX multi-process + 分布式 checkpoint）
#
# 你需要在每台机器上都启动一次本脚本，并确保：
#   - 所有机器能互相访问 MASTER_ADDR:MASTER_PORT
#   - checkpoint/data/assets 路径在多机间可见（通常需要共享文件系统）
#   - config.batch_size 必须能被“全局 GPU 总数”整除（train_dist.py 会检查）
#
# 适配 Aladdin 平台 Run Task 模式（平台会自动注入以下环境变量）：
#   nproc_per_node, nnodes, node_rank, master_addr, master_port
# 本脚本会把它们映射为 train_dist.py 需要的：
#   MASTER_ADDR, MASTER_PORT, WORLD_SIZE, WORLD_RANK
#
# 其中 nproc_per_node 通常表示“每台机器可见 GPU 数”；若你未显式设置 CUDA_VISIBLE_DEVICES，
# 本脚本会默认设置为 0..(nproc_per_node-1)。
#
# 示例（2 机，每机 8 卡）：
#   # node0
#   master_addr=10.0.0.1 master_port=12350 nnodes=2 node_rank=0 nproc_per_node=8 \
#     scripts/train_multinode_jax.sh pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k_gpu40 pi05_b1k-pt50_pretrain
#   # node1
#   master_addr=10.0.0.1 master_port=12350 nnodes=2 node_rank=1 nproc_per_node=8 \
#     scripts/train_multinode_jax.sh pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k_gpu40 pi05_b1k-pt50_pretrain
#
# 可选：
#   RUN_NORM_STATS=1  仅 rank0 先运行一次 compute_norm_stats（其它 rank 会等待它结束再开训）
#   OVERWRITE=1 / RESUME=1  传给 train_dist.py 的 --overwrite / --resume
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f ".venv/bin/activate" ]]; then
  # 与 scripts/finetune_8gpu.sh 保持一致
  source .venv/bin/activate
fi

# ---------- 配置 ----------
CONFIG_NAME="${1:-pi05_b1k-pt50_cs32_bs64_lr2.5e-5_step50k_gpu40}"
EXP_NAME="${2:-}"
if [[ $# -ge 2 ]]; then
  shift 2
elif [[ $# -ge 1 ]]; then
  shift 1
fi

# ---------- 分布式环境变量 ----------
ALADDIN_MASTER_ADDR="${master_addr:-${MASTER_ADDR:-}}"
ALADDIN_MASTER_PORT="${master_port:-${MASTER_PORT:-}}"
# On Aladdin, prefer the exported GROUP_* env vars (they are reliably set per replica).
# Some runtimes also inject non-exported shell vars like `node_rank`/`nnodes`, which may be unreliable.
ALADDIN_NNODES="${GROUP_POD_SIZE:-${nnodes:-${WORLD_SIZE:-}}}"
ALADDIN_NODE_RANK="${GROUP_POD_INDEX:-${node_rank:-${WORLD_RANK:-${RANK:-}}}}"
ALADDIN_NPROC_PER_NODE="${nproc_per_node:-${NPROC_PER_NODE:-}}"

if [[ -z "${ALADDIN_MASTER_ADDR}" ]]; then
  echo "ERROR: master_addr (or MASTER_ADDR) is required." >&2
  exit 1
fi
if [[ -z "${ALADDIN_MASTER_PORT}" ]]; then
  echo "ERROR: master_port (or MASTER_PORT) is required." >&2
  exit 1
fi
if [[ -z "${ALADDIN_NNODES}" ]]; then
  echo "ERROR: nnodes (or WORLD_SIZE) is required." >&2
  exit 1
fi
if [[ -z "${ALADDIN_NODE_RANK}" ]]; then
  echo "ERROR: node_rank (or WORLD_RANK/RANK) is required." >&2
  exit 1
fi

export MASTER_ADDR="${ALADDIN_MASTER_ADDR}"
export MASTER_PORT="${ALADDIN_MASTER_PORT}"
# 该脚本默认每机 1 个进程，因此 WORLD_SIZE=nnodes, WORLD_RANK=node_rank
export WORLD_SIZE="${ALADDIN_NNODES}"
export WORLD_RANK="${ALADDIN_NODE_RANK}"

# 进程内使用该机所有可见 GPU（逗号分隔）。若未显式设置则基于 nproc_per_node 生成。
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if [[ -n "${ALADDIN_NPROC_PER_NODE}" ]]; then
    if ! [[ "${ALADDIN_NPROC_PER_NODE}" =~ ^[0-9]+$ ]] || (( ALADDIN_NPROC_PER_NODE <= 0 )); then
      echo "ERROR: nproc_per_node must be a positive integer; got '${ALADDIN_NPROC_PER_NODE}'." >&2
      exit 1
    fi
    devs=""
    for ((i = 0; i < ALADDIN_NPROC_PER_NODE; i++)); do
      if [[ -z "${devs}" ]]; then
        devs="${i}"
      else
        devs="${devs},${i}"
      fi
    done
    export CUDA_VISIBLE_DEVICES="${devs}"
  else
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
  fi
fi

# 限制 JAX 占用的 GPU 显存比例，避免 OOM；按需调整（0.9 即 90%）
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

# 可选：关闭预分配 / 使用 platform allocator（按需开启）
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# 可选：JAX 编译缓存目录（多机多卡建议设到本地 SSD 以减少重复编译）
# export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/jax}"

if [[ -z "${EXP_NAME}" ]]; then
  if [[ "${WORLD_RANK}" == "0" ]]; then
    EXP_NAME="openpi_$(date +%Y%m%d_%H%M%S)"
  else
    # train_dist.py 会把 rank0 的 exp_name 广播到所有进程；其他 rank 传占位符即可
    EXP_NAME="openpi_placeholder"
  fi
fi

echo "CONFIG_NAME=${CONFIG_NAME}"
echo "EXP_NAME=${EXP_NAME}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "WORLD_SIZE=${WORLD_SIZE}"
echo "WORLD_RANK=${WORLD_RANK}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# # ---------- 可选：先计算 norm stats（建议只做一次） ----------
# RUN_NORM_STATS="${RUN_NORM_STATS:-0}"
# if [[ "${RUN_NORM_STATS}" == "1" ]]; then
#   if [[ "${WORLD_RANK}" == "0" ]]; then
#     echo "[rank0] Computing norm stats for ${CONFIG_NAME} ..."
#     uv run scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"
#   fi

#   # 等待 norm_stats.json 出现（需要共享文件系统）
#   NORM_STATS_FILE="$(uv run python -c "import openpi.training.config as c; cfg=c.get_config('${CONFIG_NAME}'); f=cfg.data[0] if isinstance(cfg.data, list) else cfg.data; dc=f.create(cfg.assets_dirs, cfg.model); print(str(cfg.assets_dirs / dc.repo_id / 'norm_stats.json'))")"
#   echo "[rank${WORLD_RANK}] Waiting for norm stats: ${NORM_STATS_FILE}"
#   timeout_s="${NORM_STATS_TIMEOUT_S:-7200}"
#   start_ts="$(date +%s)"
#   while [[ ! -f "${NORM_STATS_FILE}" ]]; do
#     now_ts="$(date +%s)"
#     if (( now_ts - start_ts > timeout_s )); then
#       echo "[rank${WORLD_RANK}] Timeout waiting for norm stats after ${timeout_s}s: ${NORM_STATS_FILE}" >&2
#       exit 1
#     fi
#     sleep 2
#   done
#   echo "[rank${WORLD_RANK}] Norm stats ready."
# fi

# ---------- 启动训练 ----------
TRAIN_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--overwrite)
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--resume)
fi

# 许多平台环境中 `uv run` 可能因为 cache 目录不可写而“秒退”，且日志不明显；
# 这里优先使用当前 venv 的 python 直接运行，必要时再切换回 uv。
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export JAX_TRACEBACK_FILTERING="${JAX_TRACEBACK_FILTERING:-off}"

if [[ "${USE_UV:-0}" == "1" ]]; then
  # 若确实需要 uv，可显式设置一个可写 cache 目录（例如 /tmp）以避免权限问题
  export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
  uv run scripts/train_dist.py "${CONFIG_NAME}" --exp_name="${EXP_NAME}" "${TRAIN_ARGS[@]}" "$@"
else
  # 用 venv python（若已激活则 python 即是 venv）
  python -u scripts/train_dist.py "${CONFIG_NAME}" --exp_name="${EXP_NAME}" "${TRAIN_ARGS[@]}" "$@"
fi
