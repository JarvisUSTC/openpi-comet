#!/usr/bin/env bash
#
# Stage-1: continue pretraining from openpi-comet pt50 on BEHAVIOR-1K skill-level (fine_grained_level=1)
# with stop head + prompt augmentation, while mitigating skill imbalance via probability reweighting.
#
# Target:
# - multi-node: 4 nodes * 8 GPUs/node = 32 GPUs
# - global batch size = 1024
# - peak_lr = 2.5e-5
# - warmup_steps = 3000
# - num_train_steps = 30000
# - skill reweighting derived from scripts/skill_dis.txt (downweight skills spanning >10 tasks)
#
# Usage:
#   bash scripts/aladdin/submit_multinode_skill_pretrain_pt50_resample_stop_promptaug_30k_bs1024_lr2.5e-5_warm3k.sh
#
# Optional env vars:
#   NAME, EXP_NAME, GPU_TYPE, IMAGE, CPU, MEM, WORK_DIR,
#   REPLICAS, GPU_COUNT,
#   XLA_PYTHON_CLIENT_MEM_FRACTION, USE_UV, UV_CACHE_DIR,
#   CONFIG_NAME, PT50_PARAMS_PATH, EXTRA_ARGS, DELETE_SESSION=1, DEBUG=1

set -euo pipefail

GPU_TYPE="${GPU_TYPE:-nvidia.com/gpu-h100-80gb-hbm3}"
GPU_COUNT="${GPU_COUNT:-8}"
REPLICAS="${REPLICAS:-4}"
CPU="${CPU:-16}"
MEM="${MEM:-128}"
IMAGE="${IMAGE:-registry.hd-01.alayanew.com:8443/aladdin/torch:2.6.0-cu124}"
WORK_DIR="${WORK_DIR:-/root/Training/openpi-comet-stop-head}"

CONFIG_NAME="${CONFIG_NAME:-pi05_b1k-sampled_skill_group-stop-full-pretrained-prompt-aug}"
PT50_PARAMS_PATH="${PT50_PARAMS_PATH:-/root/.cache/huggingface/hub/models--sunshk--openpi_comet/snapshots/61739ffbced89dd5ba1b87c30d93d6084b79b0af/pi05-b1kpt50-cs32/params}"

# Keep names short (Aladdin name max length is 64 chars).
EXP_NAME="${EXP_NAME:-b1k_skillpt50_resamp_stopaug_bs1024_lr2.5e-5_w3k_30k_$(date +%Y%m%d_%H%M%S)}"
NAME="${NAME:-${EXP_NAME}}"
if (( ${#NAME} > 64 )); then
  NAME="${NAME:0:64}"
fi
if (( ${#EXP_NAME} > 64 )); then
  EXP_NAME="${EXP_NAME:0:64}"
fi

XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
USE_UV="${USE_UV:-1}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

# Build skill weight list (one entry per line) from scripts/skill_dis.txt.
SKILL_ITEMS=()
SKILL_RESAMPLE_ARGS=(--tasks-threshold 10 --clip-min 0.1 --clip-max 1.0 --ref median)
if [[ -x "${WORK_DIR}/.venv/bin/python" ]]; then
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    SKILL_ITEMS+=("${line}")
  done < <("${WORK_DIR}/.venv/bin/python" "${WORK_DIR}/scripts/skill_resample_from_dis.py" "${SKILL_RESAMPLE_ARGS[@]}")
else
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    SKILL_ITEMS+=("${line}")
  done < <(python "${WORK_DIR}/scripts/skill_resample_from_dis.py" "${SKILL_RESAMPLE_ARGS[@]}")
fi

EXTRA_ARGS_DEFAULT=(
  "--data.skill-list" "${SKILL_ITEMS[@]}"
  "--batch-size" "1024"
  "--num-train-steps" "30000"
  "--lr-schedule.peak-lr" "2.5e-5"
  "--lr-schedule.warmup-steps" "3000"
  "--lr-schedule.decay-steps" "30000"
  "--weight-loader.params-path" "${PT50_PARAMS_PATH}"
)
EXTRA_ARGS="${EXTRA_ARGS:-$(printf "%q " "${EXTRA_ARGS_DEFAULT[@]}")}"

ENV_KV=(
  "CONFIG_NAME=${CONFIG_NAME}"
  "EXP_NAME=${EXP_NAME}"
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
echo "  replicas=${REPLICAS} gpu=${GPU_TYPE} x ${GPU_COUNT} per replica"
echo "  cpu=${CPU} mem_gb=${MEM}"
echo "  work_dir=${WORK_DIR}"
echo "  file=scripts/aladdin/train_multinode_place_surface_stop_promptaug.sh (generic multinode entrypoint)"
echo "  env=${ENV_STR}"
echo "  args=${EXTRA_ARGS}"

SUBMIT_LOG_DIR="${SUBMIT_LOG_DIR:-${WORK_DIR}/outputs/aladdin_submissions}"
mkdir -p "${SUBMIT_LOG_DIR}"
SUBMIT_LOG_FILE="${SUBMIT_LOG_FILE:-${SUBMIT_LOG_DIR}/${NAME}.submit.log}"

set +e
OUT="$(
  aladdin task \
    "${ALADDIN_FLAGS[@]}" \
    --name "${NAME}" \
    --image "${IMAGE}" \
    --replicas "${REPLICAS}" \
    --gpu-type "${GPU_TYPE}" \
    --gpu-count "${GPU_COUNT}" \
    --cpu "${CPU}" \
    --mem "${MEM}" \
    --work-dir "${WORK_DIR}" \
    --python /bin/bash \
    --env "${ENV_STR}" \
    --args "${EXTRA_ARGS}" \
    -f scripts/aladdin/train_multinode_place_surface_stop_promptaug.sh 2>&1 | tee "${SUBMIT_LOG_FILE}"
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
