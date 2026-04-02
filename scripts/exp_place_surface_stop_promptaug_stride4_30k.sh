#!/usr/bin/env bash
# Place-surface skill group + stop head + prompt aug
# Sampler: fixed stride (k=4), 30k steps (from 60k baseline)
#
# Usage:
#   bash scripts/exp_place_surface_stop_promptaug_stride4_30k.sh [-- extra train.py args...]
#
# Notes:
# - Uses config: pi05_b1k-sampled_skill_group-stop-full-pretrained-prompt-aug
# - Skill group: place on / place on next to / place under

set -euo pipefail

export CONFIG_NAME="${CONFIG_NAME:-pi05_b1k-sampled_skill_group-stop-full-pretrained-prompt-aug}"
export EXP_NAME="${EXP_NAME:-place_surface_stop_promptaug_stride4_30k_$(date +%Y%m%d_%H%M%S)}"

bash scripts/train_skill_group_full.sh "place_surface" "place on" "place on next to" "place under" -- \
  --num-train-steps 30000 \
  --lr-schedule.decay-steps 30000 \
  --data.frame-sampling stride \
  --data.frame-stride 4 \
  --data.frame-stride-jitter \
  --data.frame-dense-boundary-margin-frames 0 \
  "$@"
