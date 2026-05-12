#!/usr/bin/env bash
# Common Aladdin submit helper for pick_hand experiments initialized from all_skills 50k.
set -euo pipefail

: "${CONFIG_NAME:?missing CONFIG_NAME}"
: "${DEFAULT_NAME_PREFIX:?missing DEFAULT_NAME_PREFIX}"

TS="$(date +%m%d_%H%M%S)"
RESUME="${RESUME:-0}"
if [[ "${RESUME}" == "1" ]]; then
  EXP_NAME="${EXP_NAME:?missing EXP_NAME when RESUME=1}"
  NAME="${NAME:-resume_${DEFAULT_NAME_PREFIX}_${TS}}"
else
  EXP_NAME="${EXP_NAME:-${DEFAULT_NAME_PREFIX}_${TS}}"
  NAME="${NAME:-${EXP_NAME}}"
fi

if (( ${#NAME} > 64 )); then
  NAME="${NAME:0:64}"
fi

GPU_TYPE="${GPU_TYPE:-nvidia.com/gpu-h100-80gb-hbm3}"
GPU_COUNT="${GPU_COUNT:-8}"
REPLICAS="${REPLICAS:-1}"
CPU="${CPU:-16}"
MEM="${MEM:-128}"
IMAGE="${IMAGE:-registry.hd-01.alayanew.com:8443/aladdin/torch:2.6.0-cu124}"
WORK_DIR="${WORK_DIR:-/root/Training/openpi-comet}"
DELETE_SESSION="${DELETE_SESSION:-1}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
OVERWRITE="${OVERWRITE:-0}"
USE_UV="${USE_UV:-0}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

if [[ "${WANDB_ENABLED}" != "false" && "${WANDB_ENABLED}" != "0" ]]; then
  : "${WANDB_API_KEY:?WANDB_API_KEY is required when WANDB_ENABLED=true. Export it before submitting.}"
fi

ENV_KV=(
  "CONFIG_NAME=${CONFIG_NAME}"
  "EXP_NAME=${EXP_NAME}"
  "RESUME=${RESUME}"
  "OVERWRITE=${OVERWRITE}"
  "WANDB_ENABLED=${WANDB_ENABLED}"
  "WANDB_MODE=${WANDB_MODE:-online}"
  "OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/root/.cache/openpi}"
  "OPENPI_FAST_TOKENIZER_PATH=${OPENPI_FAST_TOKENIZER_PATH:-/root/Models/pi_fast_tokenizer}"
  "PYTHONPATH=${PYTHONPATH:-src:packages/openpi-client/src}"
  "XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  "XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
  "XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
  "USE_UV=${USE_UV}"
  "UV_CACHE_DIR=${UV_CACHE_DIR}"
  "PYTHONUNBUFFERED=1"
  "PYTHONFAULTHANDLER=1"
  "JAX_TRACEBACK_FILTERING=off"
)

if [[ "${WANDB_ENABLED}" != "false" && "${WANDB_ENABLED}" != "0" ]]; then
  ENV_KV+=("WANDB_API_KEY=${WANDB_API_KEY}")
fi

ENV_STR=""
ENV_STR_MASKED=""
for kv in "${ENV_KV[@]}"; do
  ENV_STR+="${kv};"
  if [[ "${kv}" == WANDB_API_KEY=* ]]; then
    ENV_STR_MASKED+="WANDB_API_KEY=***;"
  else
    ENV_STR_MASKED+="${kv};"
  fi
done

ALADDIN_FLAGS=()
if [[ "${DEBUG:-0}" == "1" ]]; then
  ALADDIN_FLAGS+=(--debug)
fi
if [[ "${DELETE_SESSION}" == "1" ]]; then
  ALADDIN_FLAGS+=(--delete-session)
fi

echo "Submitting via aladdin:"
echo "  config=${CONFIG_NAME}"
echo "  exp_name=${EXP_NAME}"
echo "  name=${NAME}"
echo "  image=${IMAGE}"
echo "  replicas=${REPLICAS} gpu=${GPU_TYPE} x ${GPU_COUNT} per replica"
echo "  cpu=${CPU} mem_gb=${MEM}"
echo "  work_dir=${WORK_DIR}"
echo "  file=scripts/aladdin/train_pick_hand_from_all_skills_50k.sh"
echo "  env=${ENV_STR_MASKED}"

SUBMIT_LOG_DIR="${SUBMIT_LOG_DIR:-${WORK_DIR}/outputs/aladdin_submissions}"
mkdir -p "${SUBMIT_LOG_DIR}"
SUBMIT_LOG_FILE="${SUBMIT_LOG_FILE:-${SUBMIT_LOG_DIR}/${NAME}.submit.log}"

TASK_ARGS=(
  --name "${NAME}"
  --image "${IMAGE}"
  --replicas "${REPLICAS}"
  --gpu-type "${GPU_TYPE}"
  --gpu-count "${GPU_COUNT}"
  --cpu "${CPU}"
  --mem "${MEM}"
  --work-dir "${WORK_DIR}"
  --python /bin/bash
  --env "${ENV_STR}"
)
TASK_ARGS+=(-f scripts/aladdin/train_pick_hand_from_all_skills_50k.sh)

set +e
OUT="$(
  aladdin task \
    "${ALADDIN_FLAGS[@]}" \
    "${TASK_ARGS[@]}" 2>&1 | tee "${SUBMIT_LOG_FILE}"
)"
STATUS="${PIPESTATUS[0]}"
set -e

TASK_ID="$(echo "${OUT}" | sed -n 's/.* id: \([0-9a-f-]\{36\}\).*/\1/p' | tail -n 1)"
if [[ -z "${TASK_ID}" && -f "${SUBMIT_LOG_FILE}" ]]; then
  TASK_ID="$(sed -n 's/.* id: \([0-9a-f-]\{36\}\).*/\1/p' "${SUBMIT_LOG_FILE}" | tail -n 1)"
fi
if [[ -n "${TASK_ID}" ]]; then
  echo "${TASK_ID}" > "${SUBMIT_LOG_DIR}/${NAME}.task_id"
  echo "Task id: ${TASK_ID} (saved to ${SUBMIT_LOG_DIR}/${NAME}.task_id)"
else
  echo "WARNING: Failed to parse task id from aladdin output. See ${SUBMIT_LOG_FILE}" >&2
fi

exit "${STATUS}"
