#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_NAME="pi05_b1k-pick_hand-20w-cosine-from-all-skills-50k" \
DEFAULT_NAME_PREFIX="pkh_20w_cos_from_all_skills_50k" \
bash "${SCRIPT_DIR}/submit_pick_hand_from_all_skills_50k_common.sh"
