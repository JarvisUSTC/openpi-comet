#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_NAME="pi05_b1k-place_into-15w-cosine-from-all-skills-50k" \
DEFAULT_NAME_PREFIX="place_into_15w_cos_from_all_skills_50k" \
bash "${SCRIPT_DIR}/submit_pick_hand_from_all_skills_50k_common.sh"
