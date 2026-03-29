#!/usr/bin/env bash
#
# Submit an evaluation job to compute stop_prob distributions on dataset batches.
#
# Usage:
#   PARAMS_DIR=outputs/checkpoints/.../<exp>/<step>/params \
#   bash scripts/aladdin/submit_eval_stop_prob_dataset.sh

set -euo pipefail

GPU_TYPE="${GPU_TYPE:-nvidia.com/gpu-h100-80gb-hbm3}"
GPU_COUNT="${GPU_COUNT:-1}"
CPU="${CPU:-8}"
MEM="${MEM:-64}"
IMAGE="${IMAGE:-registry.hd-01.alayanew.com:8443/aladdin/torch:2.6.0-cu124}"
WORK_DIR="${WORK_DIR:-/root/Training/openpi-comet-stop-head}"

CONFIG_NAME="${CONFIG_NAME:-pi05_b1k-sampled_skill_group-stop-full-pretrained}"
PARAMS_DIR="${PARAMS_DIR:?missing PARAMS_DIR (e.g. outputs/checkpoints/.../<step>/params)}"
NAME="${NAME:-eval_stop_prob_$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if ! [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] || (( GPU_COUNT <= 0 )); then
    echo "ERROR: GPU_COUNT must be a positive integer; got '${GPU_COUNT}'." >&2
    exit 1
  fi
  devs=""
  for ((i = 0; i < GPU_COUNT; i++)); do
    devs+="${devs:+,}${i}"
  done
  CUDA_VISIBLE_DEVICES="${devs}"
fi

XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

NUM_BATCHES="${NUM_BATCHES:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"

ENV_STR="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES};XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION};UV_CACHE_DIR=${UV_CACHE_DIR};PYTHONUNBUFFERED=1;PYTHONFAULTHANDLER=1;JAX_TRACEBACK_FILTERING=off;EVAL_NAME=${NAME};"
ARGS_STR="--config-name ${CONFIG_NAME} --params-dir ${PARAMS_DIR} --num-batches ${NUM_BATCHES} --batch-size ${BATCH_SIZE} --num-workers 0"
ARGS_STR+=" --skill-list \"place on:1.0\" \"place on next to:1.0\" \"place under:1.0\""

echo "Submitting eval via aladdin:"
echo "  name=${NAME}"
echo "  params_dir=${PARAMS_DIR}"
echo "  args=${ARGS_STR}"

aladdin task \
  --name "${NAME}" \
  --image "${IMAGE}" \
  --gpu-type "${GPU_TYPE}" \
  --gpu-count "${GPU_COUNT}" \
  --cpu "${CPU}" \
  --mem "${MEM}" \
  --work-dir "${WORK_DIR}" \
  --python /bin/bash \
  --env "${ENV_STR}" \
  --args "${ARGS_STR}" \
  -f scripts/aladdin/eval_stop_prob_dataset.sh
