#!/usr/bin/env bash
# Similar ops: pick / hold / release / hand over
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

bash "${REPO_ROOT}/scripts/train_skill_group_lora.sh" \
  "pick_handover" \
  "pick up from" \
  "hold" \
  "release" \
  "hand over" \
  -- "$@"

