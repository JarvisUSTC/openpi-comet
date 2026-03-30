#!/usr/bin/env bash
#
# Submit a single-node multi-GPU Pi05 (place_surface group) training job WITH stop supervision via `aladdin`.
#
# Usage:
#   GPU_COUNT=8 \
#   EXTRA_ARGS="--num-train-steps 2000" \
#   bash scripts/aladdin/submit_group_place_surface_stop.sh
#
# Optional env vars:
#   NAME, EXP_NAME, GPU_TYPE, IMAGE, CPU, MEM, WORK_DIR,
#   CUDA_VISIBLE_DEVICES, XLA_PYTHON_CLIENT_MEM_FRACTION, USE_UV, UV_CACHE_DIR,
#   CONFIG_NAME, EXTRA_ARGS, DELETE_SESSION=1, DEBUG=1

set -euo pipefail

GPU_TYPE="${GPU_TYPE:-nvidia.com/gpu-h100-80gb-hbm3}"
GPU_COUNT="${GPU_COUNT:-8}"
CPU="${CPU:-16}"
MEM="${MEM:-128}"
IMAGE="${IMAGE:-registry.hd-01.alayanew.com:8443/aladdin/torch:2.6.0-cu124}"
WORK_DIR="${WORK_DIR:-/root/Training/openpi-comet-stop-head}"

CONFIG_NAME="${CONFIG_NAME:-pi05_b1k-sampled_skill_group-stop-full-pretrained}"
EXP_NAME="${EXP_NAME:-place_surface_stop_$(date +%Y%m%d_%H%M%S)}"
NAME="${NAME:-${EXP_NAME}}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if ! [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] || (( GPU_COUNT <= 0 )); then
    echo "ERROR: GPU_COUNT must be a positive integer; got '${GPU_COUNT}'." >&2
    exit 1
  fi
  devs=""
  for ((i = 0; i < GPU_COUNT; i++)); do
    if [[ -z "${devs}" ]]; then
      devs="${i}"
    else
      devs="${devs},${i}"
    fi
  done
  CUDA_VISIBLE_DEVICES="${devs}"
fi

XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
USE_UV="${USE_UV:-1}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

ENV_KV=(
  "CONFIG_NAME=${CONFIG_NAME}"
  "EXP_NAME=${EXP_NAME}"
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  "XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION}"
  "USE_UV=${USE_UV}"
  "UV_CACHE_DIR=${UV_CACHE_DIR}"
  "PYTHONUNBUFFERED=1"
  "PYTHONFAULTHANDLER=1"
  "JAX_TRACEBACK_FILTERING=off"
)

ENV_STR=""
for kv in "${ENV_KV[@]}"; do
  ENV_STR+="${kv};"
done

ARGS_STR=""
if [[ -n "${EXTRA_ARGS}" ]]; then
  ARGS_STR="${EXTRA_ARGS}"
fi

ALADDIN_FLAGS=()
if [[ "${DEBUG:-0}" == "1" ]]; then
  ALADDIN_FLAGS+=(--debug)
fi
if [[ "${DELETE_SESSION:-1}" == "1" ]]; then
  ALADDIN_FLAGS+=(--delete-session)
fi

echo "Submitting via aladdin:"
echo "  name=${NAME}"
echo "  image=${IMAGE}"
echo "  gpu=${GPU_TYPE} x ${GPU_COUNT}"
echo "  cpu=${CPU} mem_gb=${MEM}"
echo "  work_dir=${WORK_DIR}"
echo "  file=scripts/aladdin/train_group_place_surface_stop.sh (via /bin/bash)"
echo "  env=${ENV_STR}"
if [[ -n "${ARGS_STR}" ]]; then
  echo "  args=${ARGS_STR}"
fi

SUBMIT_LOG_DIR="${SUBMIT_LOG_DIR:-${WORK_DIR}/outputs/aladdin_submissions}"
mkdir -p "${SUBMIT_LOG_DIR}"
SUBMIT_LOG_FILE="${SUBMIT_LOG_FILE:-${SUBMIT_LOG_DIR}/${NAME}.submit.log}"

set +e
OUT="$(
  aladdin task \
    "${ALADDIN_FLAGS[@]}" \
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
    -f scripts/aladdin/train_group_place_surface_stop.sh 2>&1 | tee "${SUBMIT_LOG_FILE}"
)"
STATUS="${PIPESTATUS[0]}"
set -e

TASK_ID="$(echo "${OUT}" | sed -n 's/.* id: \\([0-9a-f-]\\{36\\}\\).*/\\1/p' | tail -n 1)"
if [[ -z "${TASK_ID}" && -f "${SUBMIT_LOG_FILE}" ]]; then
  TASK_ID="$(sed -n 's/.* id: \\([0-9a-f-]\\{36\\}\\).*/\\1/p' "${SUBMIT_LOG_FILE}" | tail -n 1)"
fi
if [[ -n "${TASK_ID}" ]]; then
  echo "${TASK_ID}" > "${SUBMIT_LOG_DIR}/${NAME}.task_id"
  echo "Task id: ${TASK_ID} (saved to ${SUBMIT_LOG_DIR}/${NAME}.task_id)"
else
  echo "WARNING: Failed to parse task id from aladdin output. See ${SUBMIT_LOG_FILE}" >&2
fi

exit "${STATUS}"
