#!/usr/bin/env bash
# Similar ops: placing on/under/next-to (spatial relations on surfaces)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

bash "${REPO_ROOT}/scripts/train_skill_group_lora.sh" \
  "place_surface" \
  "place on" \
  "place on next to" \
  "place under" \
  -- "$@"

