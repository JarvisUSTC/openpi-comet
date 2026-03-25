#!/usr/bin/env bash
# Similar ops: button/switch actuation
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

bash "${REPO_ROOT}/scripts/train_skill_group_lora.sh" \
  "switch_press" \
  "turn on switch" \
  "turn off switch" \
  "press" \
  -- "$@"

