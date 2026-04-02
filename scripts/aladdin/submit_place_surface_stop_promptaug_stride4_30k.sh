#!/usr/bin/env bash
#
# Submit: place_surface + stop head + prompt aug
# Sampler: fixed stride=4 (~1/4 frames), steps: 60k -> 30k
#
# Usage:
#   bash scripts/aladdin/submit_place_surface_stop_promptaug_stride4_30k.sh
#
# You can override any env vars supported by `scripts/aladdin/submit_group_place_surface_stop.sh`
# (GPU_TYPE/GPU_COUNT/IMAGE/WORK_DIR/...), plus:
#   EXP_NAME, NAME, EXTRA_ARGS

set -euo pipefail

export CONFIG_NAME="${CONFIG_NAME:-pi05_b1k-sampled_skill_group-stop-full-pretrained-prompt-aug}"
export EXP_NAME="${EXP_NAME:-place_surface_stop_promptaug_stride4_30k_$(date +%Y%m%d_%H%M%S)}"
export NAME="${NAME:-${EXP_NAME}}"

EXTRA_ARGS_DEFAULT=(
  "--num-train-steps" "30000"
  "--lr-schedule.decay-steps" "30000"
  "--data.frame-sampling" "stride"
  "--data.frame-stride" "4"
  "--data.frame-stride-jitter"
)
if [[ -z "${EXTRA_ARGS:-}" ]]; then
  EXTRA_ARGS="$(printf "%q " "${EXTRA_ARGS_DEFAULT[@]}")"
fi
export EXTRA_ARGS

bash scripts/aladdin/submit_group_place_surface_stop.sh
