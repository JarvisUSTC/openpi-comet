#!/usr/bin/env bash
# Similar ops: tray manipulation (push/pull)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

bash "${REPO_ROOT}/scripts/train_skill_group_lora.sh" \
  "tray" \
  "push tray" \
  "pull tray" \
  -- "$@"

