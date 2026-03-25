#!/usr/bin/env bash
# Train a full-parameter model on a group of similar skills (union filter).
#
# Usage:
#   scripts/train_skill_group_full.sh "<group_name>" "<skill 1>" ["<skill 2>" ...] [-- extra train.py args...]
#
# Examples:
#   bash scripts/train_skill_group_full.sh "open_close" "open door" "close door" "open drawer" "close drawer"
#   CONFIG_NAME=pi05_b1k-sampled_skill_group-full bash scripts/train_skill_group_full.sh "nav" "move to" "turn to" -- --num-train-steps 200
#
# Notes:
# - behavior dataset expects skill_list entries like "<skill>:<weight>" (see src/behavior/learning/datas/dataset.py).
# - This script uses a filter-only setup by default: skills not listed will be rejected.

set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.venv/bin/activate"
fi

# Avoid torch inductor spawning compile worker subprocesses in restricted environments.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

GROUP="${1:?missing <group_name>}"
shift 1

SKILLS=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--" ]]; then
    shift 1
    break
  fi
  SKILLS+=("$1")
  shift 1
done

if [[ ${#SKILLS[@]} -eq 0 ]]; then
  echo "No skills provided. Usage: scripts/train_skill_group_full.sh \"group\" \"skill1\" [\"skill2\" ...] [-- extra args]" >&2
  exit 2
fi

CONFIG_NAME="${CONFIG_NAME:-pi05_b1k-sampled_skill_group-full}"
export CONFIG_NAME="${CONFIG_NAME}"
EXP_NAME="${EXP_NAME:-pretrain_full_${GROUP//[^a-zA-Z0-9_-]/_}_$(date +%Y%m%d_%H%M%S)}"

SKILL_WEIGHT="${SKILL_WEIGHT:-1.0}"
SKILL_ITEMS=()
if [[ ${#SKILLS[@]} -eq 1 && "${SKILLS[0]}" == "all" ]]; then
  # Special case: "all" means no skill filtering. The dataset expects the literal "all",
  # not a weighted entry like "all:1.0".
  SKILL_ITEMS=("all")
else
  for s in "${SKILLS[@]}"; do
    if [[ "${s}" == "all" ]]; then
      echo "Invalid skill list: cannot mix \"all\" with other skills. Got: ${SKILLS[*]}" >&2
      exit 2
    fi
    SKILL_ITEMS+=("${s}:${SKILL_WEIGHT}")
  done
fi

echo "Config: ${CONFIG_NAME}"
echo "Group: ${GROUP}"
echo "Skills: ${SKILLS[*]}"
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
    python scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"
  fi
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

python -u scripts/train.py "${CONFIG_NAME}" \
  --exp-name="${EXP_NAME}" \
  --overwrite \
  --data.skill-list "${SKILL_ITEMS[@]}" \
  "$@"
