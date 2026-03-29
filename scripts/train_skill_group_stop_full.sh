#!/usr/bin/env bash
# Train a full-parameter Pi05 model on a group of similar skills (union filter) WITH weak stop supervision.
#
# Usage:
#   scripts/train_skill_group_stop_full.sh "<group_name>" "<skill 1>" ["<skill 2>" ...] [-- extra train.py args...]
#
# Examples:
#   bash scripts/train_skill_group_stop_full.sh "place_surface" "place on" "place on next to" "place under"
#   CONFIG_NAME=pi05_b1k-sampled_skill_group-stop-full-pretrained bash scripts/train_skill_group_stop_full.sh "nav" "move to" "turn to" -- --num-train-steps 200
#
# Notes:
# - This script is a thin wrapper over `scripts/train_skill_group_full.sh`; "stop" behavior comes from the config.

set -euo pipefail

# Default to the stop-supervision config, but allow override.
export CONFIG_NAME="${CONFIG_NAME:-pi05_b1k-sampled_skill_group-stop-full-pretrained}"

bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_skill_group_full.sh" "$@"

