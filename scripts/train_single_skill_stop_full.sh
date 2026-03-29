#!/usr/bin/env bash
# Train a full-parameter Pi05 model on a single skill with weak stop supervision.
#
# Usage:
#   bash scripts/train_single_skill_stop_full.sh "<skill_name>" [-- extra train.py args...]
#
# Examples:
#   bash scripts/train_single_skill_stop_full.sh "open door"
#   bash scripts/train_single_skill_stop_full.sh "move to" -- --num-train-steps 200
#
# Notes:
# - behavior dataset expects skill_list entries like "<skill>:<weight>" (see src/behavior/learning/datas/dataset.py).
# - Default config can be overridden via CONFIG_NAME env var.

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

SKILL="${1:?missing <skill_name>}"
shift 1

# Allow passing extra args after `--` for consistency with other scripts.
if [[ $# -gt 0 && "$1" == "--" ]]; then
  shift 1
fi

CONFIG_NAME="${CONFIG_NAME:-pi05_b1k-sampled_single_skill-stop-full-pretrained}"
EXP_NAME="${EXP_NAME:-stop_full_${SKILL//[^a-zA-Z0-9_-]/_}_$(date +%Y%m%d_%H%M%S)}"

SKILL_WEIGHT="${SKILL_WEIGHT:-1.0}"
SKILL_ITEM="${SKILL}:${SKILL_WEIGHT}"

echo "Config: ${CONFIG_NAME}"
echo "Skill: ${SKILL}"
echo "Skill weight: ${SKILL_WEIGHT}"
echo "exp_name: ${EXP_NAME}"

# Compute norm stats once for this config if missing (can skip with SKIP_NORM_STATS=1).
if [[ "${SKIP_NORM_STATS:-0}" != "1" ]]; then
  NORM_STATS_PATH="$(python - <<'PY'
import os
import openpi.training.config as c

cfg = c.get_config(os.environ["CONFIG_NAME"])
factory = cfg.data[0] if isinstance(cfg.data, list) else cfg.data
repo_id = getattr(factory, "repo_id", None)
print(cfg.assets_dirs / repo_id / "norm_stats.json")
PY
)"
  if [[ ! -f "${NORM_STATS_PATH}" ]]; then
    echo "Norm stats not found at: ${NORM_STATS_PATH}"
    echo "Computing norm stats..."
    uv run scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"
  fi
fi

uv run scripts/train.py "${CONFIG_NAME}" \
  --exp-name="${EXP_NAME}" \
  --data.skill-list "${SKILL_ITEM}" \
  "$@"

