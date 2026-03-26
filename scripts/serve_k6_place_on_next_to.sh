#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Use behavior-comet conda env (has omnigibson)
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate behavior-comet 2>/dev/null || true

python scripts/serve_b1k.py \
  policy:checkpoint \
  --policy.config pi05_b1k-sampled_single_skill-full \
  --policy.dir "${CHECKPOINT_DIR:-$HOME/Jiawei/openpi-comet-baseline/checkpoints/params}" \
  --task-name "${TASK_NAME:-place_on_next_to}" \
  --port "${PORT:-8000}"
