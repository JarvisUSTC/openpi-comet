#!/usr/bin/env bash
# Single skill: place on next to
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Right-click friendly defaults for terminal-weighted fine-tuning from the old 145k checkpoint.
DEFAULT_EXP_NAME="pretrain_full_place_on_next_to_terminal_weighted_$(date +%Y%m%d_%H%M%S)"
RESUME="${RESUME:-0}"
EXP_NAME="${EXP_NAME:-${DEFAULT_EXP_NAME}}"
INIT_SOURCE_EXP_NAME="${INIT_SOURCE_EXP_NAME:-pretrain_full_place_on_next_to_20260329_004955}"
INIT_STEP="${INIT_STEP:-145000}"
INIT_CHECKPOINT_ROOT="${INIT_CHECKPOINT_ROOT:-/root/Training/openpi-comet-baseline/outputs/checkpoints/pi05_b1k-sampled_single_skill-full}"
INIT_PARAMS_PATH="${INIT_PARAMS_PATH:-${INIT_CHECKPOINT_ROOT}/${INIT_SOURCE_EXP_NAME}/${INIT_STEP}/params}"

if [[ ! -e "${INIT_PARAMS_PATH}" ]]; then
  echo "Init checkpoint not found: ${INIT_PARAMS_PATH}" >&2
  echo "Override with INIT_PARAMS_PATH=/abs/path/to/params or set INIT_SOURCE_EXP_NAME / INIT_STEP." >&2
  exit 1
fi

export OPENPI_INIT_PARAMS_PATH="${INIT_PARAMS_PATH}"

CONFIG_NAME=pi05_b1k-sampled_single_skill-full \
OPENPI_INIT_PARAMS_PATH="${OPENPI_INIT_PARAMS_PATH}" \
RESUME="${RESUME}" \
EXP_NAME="${EXP_NAME}" \
bash "${REPO_ROOT}/scripts/train_skill_group_full.sh" \
  "place_on_next_to" \
  "place on next to" \
  -- "$@"
