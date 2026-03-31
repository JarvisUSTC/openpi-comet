#!/usr/bin/env bash
# Skill group: place_surface (+ stop supervision)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_NAME="${CONFIG_NAME:-pi05_b1k-sampled_skill_group-stop-full-pretrained}" \
bash "${REPO_ROOT}/scripts/train_skill_group_stop_full.sh" \
  "place_surface" \
  "place on" \
  "place on next to" \
  "place under" \
  -- "$@"
