#!/usr/bin/env bash
# Skill group: place_into
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_NAME=pi05_b1k-sampled_skill_group-full \
bash "${REPO_ROOT}/scripts/train_skill_group_full.sh" \
  "place_into" \
  "place in" \
  "place in next to" \
  "insert" \
  -- "$@"
