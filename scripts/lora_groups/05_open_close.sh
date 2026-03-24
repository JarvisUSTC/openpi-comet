#!/usr/bin/env bash
# Similar ops: open/close articulated objects (door/drawer/lid)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

bash "${REPO_ROOT}/scripts/train_skill_group_lora.sh" \
  "open_close" \
  "open door" \
  "close door" \
  "open drawer" \
  "close drawer" \
  "open lid" \
  "close lid" \
  -- "$@"

