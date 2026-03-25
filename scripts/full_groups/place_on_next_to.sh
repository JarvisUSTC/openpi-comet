#!/usr/bin/env bash
# Single skill: place on next to
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_NAME=pi05_b1k-sampled_single_skill-full \
bash "${REPO_ROOT}/scripts/train_skill_group_full.sh" \
  "place_on_next_to" \
  "place on next to" \
  -- "$@"
