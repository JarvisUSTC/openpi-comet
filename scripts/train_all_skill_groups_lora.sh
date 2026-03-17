#!/usr/bin/env bash
# Train LoRA models for all predefined skill groups.
#
# Default: sequential execution (one finishes, then the next starts).
# You can pass extra train args after `--`, they will be forwarded to every group.
#
# Example:
#   bash scripts/train_all_skill_groups_lora.sh -- --num-train-steps 200 --save-interval 200
#
# Common env overrides:
#   CONFIG_NAME=pi05_b1k-sampled_single_skill-lora
#   SKILL_WEIGHT=1.0
#   WANDB_ENABLED=1
#   SKIP_NORM_STATS=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

EXTRA_ARGS=()
if [[ "${1:-}" == "--" ]]; then
  shift 1
  EXTRA_ARGS=("$@")
fi

GROUP_SCRIPTS=(
  "scripts/lora_groups/01_nav_orient.sh"
  "scripts/lora_groups/02_pick_handover.sh"
  "scripts/lora_groups/03_place_into.sh"
  "scripts/lora_groups/04_place_surface.sh"
  "scripts/lora_groups/05_open_close.sh"
  "scripts/lora_groups/06_switch_press.sh"
  "scripts/lora_groups/07_cleaning.sh"
  "scripts/lora_groups/08_tray.sh"
  "scripts/lora_groups/09_liquid_spray_tip.sh"
  "scripts/lora_groups/10_cut_fire.sh"
  "scripts/lora_groups/11_attach_hang.sh"
)

for s in "${GROUP_SCRIPTS[@]}"; do
  echo "================================================================================"
  echo "Running: ${s}"
  echo "================================================================================"
  bash "${REPO_ROOT}/${s}" -- "${EXTRA_ARGS[@]}"
done

