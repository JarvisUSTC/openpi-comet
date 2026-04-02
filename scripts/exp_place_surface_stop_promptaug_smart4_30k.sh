#!/usr/bin/env bash
# Place-surface skill group + stop head + prompt aug
# Sampler: smart (base stride=4) while keeping ~1/4 sampling, 30k steps (from 60k baseline)
#
# Usage:
#   bash scripts/exp_place_surface_stop_promptaug_smart4_30k.sh [-- extra train.py args...]
#
# Smart sampling design (lightweight to keep sampling close to 1/4):
# - Base stride=4 with random phase per sampled range
# - Dense sampling near segment boundaries (small margin)
# - Densify around:
#   - 1 largest overall action-change event
#   - 1 largest gripper-change event (dims default to [14,22] for R1-Pro when not specified)

set -euo pipefail

export CONFIG_NAME="${CONFIG_NAME:-pi05_b1k-sampled_skill_group-stop-full-pretrained-prompt-aug}"
export EXP_NAME="${EXP_NAME:-place_surface_stop_promptaug_smart4_30k_$(date +%Y%m%d_%H%M%S)}"

bash scripts/train_skill_group_full.sh "place_surface" "place on" "place on next to" "place under" -- \
  --num-train-steps 30000 \
  --lr-schedule.decay-steps 30000 \
  --data.frame-sampling smart \
  --data.frame-stride 4 \
  --data.frame-stride-jitter \
  --data.frame-dense-boundary-margin-frames 4 \
  --data.smart-num-action-events 1 \
  --data.smart-action-event-window-frames 1 \
  --data.smart-contact-num-events 1 \
  --data.smart-contact-window-frames 1 \
  "$@"
