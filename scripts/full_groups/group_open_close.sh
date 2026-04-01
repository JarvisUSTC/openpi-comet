#!/usr/bin/env bash
# Skill group: open_close
git branch --show-current
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_NAME=pi05_b1k-sampled_skill_group-full \
bash "${REPO_ROOT}/scripts/train_skill_group_full.sh" \
  "open_close" \
  "open door" \
  "close door" \
  "open drawer" \
  "close drawer" \
  "open lid" \
  "close lid" \
  -- "$@"


# CONFIG_NAME=pi05_b1k-sampled_skill_group-full \
# EXP_NAME=pretrain_full_open_close_20260325_001248 \
# SKIP_NORM_STATS=1 \
# bash "${REPO_ROOT}/scripts/train_skill_group_full.sh" \
#   "open_close" \
#   "open door" \
#   "close door" \
#   "open drawer" \
#   "close drawer" \
#   "open lid" \
#   "close lid" \
#   -- --resume
