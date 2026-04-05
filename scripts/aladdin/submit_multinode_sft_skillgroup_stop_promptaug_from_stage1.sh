#!/usr/bin/env bash
#
# Stage-2: SFT on a skill group (filtered) starting from a Stage-1 checkpoint.
#
# Defaults:
# - multi-node: 2 nodes * 8 GPUs/node = 16 GPUs
# - global batch size = 512
# - peak_lr = 2.5e-6
# - stop head + prompt augmentation enabled (via config)
#
# Usage:
#   STAGE1_PARAMS_PATH=/path/to/stage1/params \
#     bash scripts/aladdin/submit_multinode_sft_skillgroup_stop_promptaug_from_stage1.sh <group>
#
# Groups:
#   - nav_motion     (30k steps)
#   - pick_hand      (30k steps)
#   - place          (30k steps)
#   - open_close     (30k steps)
#   - switch_control (10k steps)
#   - tool_clean     (10k steps)
#
# Optional env vars:
#   NAME, EXP_NAME, GPU_TYPE, IMAGE, CPU, MEM, WORK_DIR,
#   REPLICAS, GPU_COUNT,
#   XLA_PYTHON_CLIENT_MEM_FRACTION, USE_UV, UV_CACHE_DIR,
#   CONFIG_NAME, EXTRA_ARGS, DELETE_SESSION=1, DEBUG=1

set -euo pipefail

GROUP="${1:?missing <group> (e.g. place)}"

GPU_TYPE="${GPU_TYPE:-nvidia.com/gpu-h100-80gb-hbm3}"
GPU_COUNT="${GPU_COUNT:-8}"
REPLICAS="${REPLICAS:-2}"
CPU="${CPU:-16}"
MEM="${MEM:-128}"
IMAGE="${IMAGE:-registry.hd-01.alayanew.com:8443/aladdin/torch:2.6.0-cu124}"
WORK_DIR="${WORK_DIR:-/root/Training/openpi-comet-stop-head}"

CONFIG_NAME="${CONFIG_NAME:-pi05_b1k-sampled_skill_group-stop-full-pretrained-prompt-aug}"
STAGE1_PARAMS_PATH="${STAGE1_PARAMS_PATH:-}"
if [[ -z "${STAGE1_PARAMS_PATH}" ]]; then
  echo "ERROR: STAGE1_PARAMS_PATH is required (path to .../params directory)." >&2
  exit 2
fi

SKILL_ITEMS=()
NUM_TRAIN_STEPS=""

case "${GROUP}" in
  nav_motion)
    SKILL_ITEMS=("move to" "turn to" "push to" "pull tray" "push tray")
    NUM_TRAIN_STEPS="30000"
    ;;
  pick_hand)
    SKILL_ITEMS=("pick up from" "hold" "release" "hand over")
    NUM_TRAIN_STEPS="30000"
    ;;
  place)
    SKILL_ITEMS=("place on" "place on next to" "place in" "place in next to" "place under")
    NUM_TRAIN_STEPS="30000"
    ;;
  open_close)
    SKILL_ITEMS=("open door" "close door" "open lid" "close lid" "open drawer" "close drawer")
    NUM_TRAIN_STEPS="30000"
    ;;
  switch_control)
    SKILL_ITEMS=("turn on switch" "turn off switch" "press" "ignite")
    NUM_TRAIN_STEPS="10000"
    ;;
  tool_clean)
    SKILL_ITEMS=("sweep surface" "sweep off" "wipe hard" "spray" "pour" "chop" "insert" "attach" "hang" "tip over")
    NUM_TRAIN_STEPS="10000"
    ;;
  *)
    echo "ERROR: Unknown group '${GROUP}'." >&2
    exit 2
    ;;
esac

WARMUP_STEPS="$(( NUM_TRAIN_STEPS / 10 ))"
if (( WARMUP_STEPS < 500 )); then
  WARMUP_STEPS=500
fi
if (( WARMUP_STEPS > 3000 )); then
  WARMUP_STEPS=3000
fi

# Keep names short (Aladdin name max length is 64 chars).
EXP_NAME="${EXP_NAME:-sft_${GROUP}_bs512_lr2.5e-6_${NUM_TRAIN_STEPS}_$(date +%Y%m%d_%H%M%S)}"
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

EXTRA_ARGS_DEFAULT=(
  "--data.skill-list" "${SKILL_ITEMS[@]}"
  "--batch-size" "512"
  "--num-train-steps" "${NUM_TRAIN_STEPS}"
  "--lr-schedule.peak-lr" "2.5e-6"
  "--lr-schedule.warmup-steps" "${WARMUP_STEPS}"
  "--lr-schedule.decay-steps" "${NUM_TRAIN_STEPS}"
  "--weight-loader.params-path" "${STAGE1_PARAMS_PATH}"
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
echo "  group=${GROUP} steps=${NUM_TRAIN_STEPS} warmup=${WARMUP_STEPS}"
echo "  skills=$(printf "%q " "${SKILL_ITEMS[@]}")"
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
