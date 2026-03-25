#!/usr/bin/env bash
# Similar ops: kitchen-style cutting / ignition
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

bash "${REPO_ROOT}/scripts/train_skill_group_lora.sh" \
  "cut_fire" \
  "chop" \
  "ignite" \
  -- "$@"

