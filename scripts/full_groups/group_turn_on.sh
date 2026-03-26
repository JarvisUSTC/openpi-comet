#!/usr/bin/env bash
# Single skill: turn on switch
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# CONFIG_NAME=pi05_b1k-sampled_skill_group-full \
# bash "${REPO_ROOT}/scripts/train_skill_group_full.sh" \
#   "switch_press" \
#   "turn on switch" \
#   "turn off switch" \
#   "press" \
#   -- "$@"

CONFIG_NAME=pi05_b1k-sampled_skill_group-full \
EXP_NAME=pretrain_full_switch_press_20260323_171824 \
SKIP_NORM_STATS=1 \
bash "${REPO_ROOT}/scripts/train_skill_group_full.sh" \
  "switch_press" \
  "turn on switch" \
  "turn off switch" \
  "press" \
  -- --resume
