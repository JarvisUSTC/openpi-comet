#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ===== User-configurable variables =====
VENV_DIR="${VENV_DIR:-/vepfs-C/hyn/openpi-codebase/.venv}"
OPENPI_DATA_HOME_DIR="${OPENPI_DATA_HOME_DIR:-/b1k/.cache/openpi}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0}"

CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/outputs/checkpoints/pi05_b1k-ki-vqa-joint-pick-up-from-no-task-planning/pi05_ki_joint_pick_up_from_vqa_no_task_planning_ga1/5000}"
# Example alternatives:
# CKPT_DIR="/dataset-vla/checkpoints/pi05_b1k-all_skills/49999"
# CKPT_DIR="/b1k/model_pretrained/pi/pi05_base"

SAMPLES_FILE="${SAMPLES_FILE:-${REPO_ROOT}/vqa_eval/data/attn_samples_dir.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/vqa_eval/outputs}"
CKPT_NAME="$(basename "${CKPT_DIR}")"
RUN_NAME="${RUN_NAME:-attn_${CKPT_NAME}}"

# Optional: leave empty to run all samples.
LIMIT="${LIMIT:-}"

# ===== Basic checks =====
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "Repository root not found: ${REPO_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "Virtual environment activate script not found: ${VENV_DIR}/bin/activate" >&2
  exit 1
fi

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "Checkpoint directory not found: ${CKPT_DIR}" >&2
  exit 1
fi

if [[ ! -d "${CKPT_DIR}/params" ]]; then
  echo "Checkpoint params directory not found: ${CKPT_DIR}/params" >&2
  exit 1
fi

if [[ ! -f "${SAMPLES_FILE}" ]]; then
  echo "Samples file not found: ${SAMPLES_FILE}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
TIMESTAMP_DIR_NAME="$(date +%Y%m%d_%H%M%S)"
TIMESTAMP_DIR_PATH="${OUTPUT_ROOT}/${TIMESTAMP_DIR_NAME}"
TEMP_OUTPUT_ROOT="${TIMESTAMP_DIR_PATH}/_extract_root"
SOURCE_ATTN_DIR_PATH=""
RENAMED_RUN_DIR_PATH="${TIMESTAMP_DIR_PATH}/${RUN_NAME}"

while [[ -e "${TIMESTAMP_DIR_PATH}" ]]; do
  sleep 1
  TIMESTAMP_DIR_NAME="$(date +%Y%m%d_%H%M%S)"
  TIMESTAMP_DIR_PATH="${OUTPUT_ROOT}/${TIMESTAMP_DIR_NAME}"
  TEMP_OUTPUT_ROOT="${TIMESTAMP_DIR_PATH}/_extract_root"
  RENAMED_RUN_DIR_PATH="${TIMESTAMP_DIR_PATH}/${RUN_NAME}"
done

mkdir -p "${TIMESTAMP_DIR_PATH}"
mkdir -p "${TEMP_OUTPUT_ROOT}"

echo "Running attention extraction with:"
echo "  REPO_ROOT=${REPO_ROOT}"
echo "  VENV_DIR=${VENV_DIR}"
echo "  OPENPI_DATA_HOME=${OPENPI_DATA_HOME_DIR}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}"
echo "  CKPT_DIR=${CKPT_DIR}"
echo "  SAMPLES_FILE=${SAMPLES_FILE}"
echo "  OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "  TIMESTAMP_DIR=${TIMESTAMP_DIR_PATH}"
echo "  TEMP_OUTPUT_ROOT=${TEMP_OUTPUT_ROOT}"
echo "  RUN_NAME=${RUN_NAME}"
echo "  RENDER=1"
if [[ -n "${TOKENIZER_PATH}" ]]; then
  echo "  TOKENIZER_PATH=${TOKENIZER_PATH}"
fi
if [[ -n "${LIMIT}" ]]; then
  echo "  LIMIT=${LIMIT}"
else
  echo "  LIMIT=<all samples>"
fi

cd "${REPO_ROOT}"
source "${VENV_DIR}/bin/activate"

CMD=(
  python
  "vqa_eval/scripts/extract_attn.py"
  "--ckpt" "${CKPT_DIR}"
  "--samples" "${SAMPLES_FILE}"
  "--out-dir" "${TEMP_OUTPUT_ROOT}"
)

if [[ -n "${TOKENIZER_PATH}" ]]; then
  CMD+=("--tokenizer" "${TOKENIZER_PATH}")
fi

if [[ -n "${LIMIT}" ]]; then
  CMD+=("--limit" "${LIMIT}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
OPENPI_DATA_HOME="${OPENPI_DATA_HOME_DIR}" \
"${CMD[@]}"

EXTRACT_TIMESTAMP_DIR="$(find "${TEMP_OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
if [[ -z "${EXTRACT_TIMESTAMP_DIR}" ]]; then
  echo "Could not find extract_attn output directory under ${TEMP_OUTPUT_ROOT}" >&2
  exit 1
fi

SOURCE_ATTN_DIR_PATH="${EXTRACT_TIMESTAMP_DIR}/attn"
if [[ ! -d "${SOURCE_ATTN_DIR_PATH}" ]]; then
  echo "Expected attention output directory not found: ${SOURCE_ATTN_DIR_PATH}" >&2
  exit 1
fi

if [[ -e "${RENAMED_RUN_DIR_PATH}" ]]; then
  echo "Target run directory already exists: ${RENAMED_RUN_DIR_PATH}" >&2
  exit 1
fi

mv "${SOURCE_ATTN_DIR_PATH}" "${RENAMED_RUN_DIR_PATH}"
rm -rf "${TEMP_OUTPUT_ROOT}"

RENDER_CMD=(
  python
  "vqa_eval/scripts/render_attn.py"
  "--attn-dir" "${RENAMED_RUN_DIR_PATH}"
)

"${RENDER_CMD[@]}"

echo
echo "Finished."
echo "Results directory: ${RENAMED_RUN_DIR_PATH}"
