#!/usr/bin/env bash
# Similar ops: navigation / orientation / pushing motion primitives
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

bash "${REPO_ROOT}/scripts/train_skill_group_lora.sh" \
  "nav_orient" \
  "move to" \
  "turn to" \
  "push to" \
  -- "$@"

