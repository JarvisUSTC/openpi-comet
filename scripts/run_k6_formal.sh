#!/usr/bin/env bash
set -euo pipefail

# K=6 formal training: 50k steps with video memory.
# Usage:
#   bash scripts/run_k6_formal.sh
# Optional env overrides:
#   OVERWRITE=true
#   WANDB_ENABLED=true

CONFIG_NAME="${CONFIG_NAME:-pi05_b1k-k6-smoke-step50000}"
EXP_NAME="${EXP_NAME:-k6_formal_$(date +%Y%m%d_%H%M%S)}"
OVERWRITE="${OVERWRITE:-true}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] Config: ${CONFIG_NAME}"
echo "[INFO] Exp name: ${EXP_NAME}"
echo "[INFO] Overwrite: ${OVERWRITE}"
echo "[INFO] Wandb enabled: ${WANDB_ENABLED}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[INFO] GPU status:"
  nvidia-smi || true
fi

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

EXTRA_ARGS=()
if [[ "${OVERWRITE}" == "true" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi
if [[ "${WANDB_ENABLED}" == "true" ]]; then
  EXTRA_ARGS+=(--wandb-enabled)
else
  EXTRA_ARGS+=(--no-wandb-enabled)
fi

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  TRAIN_CMD=("${REPO_ROOT}/.venv/bin/python")
elif command -v python >/dev/null 2>&1; then
  TRAIN_CMD=(python)
elif command -v uv >/dev/null 2>&1; then
  TRAIN_CMD=(uv run python)
else
  echo "[ERROR] Neither .venv python, system python, nor uv is available." >&2
  exit 1
fi

echo "[INFO] Starting training..."
echo "${TRAIN_CMD[*]} scripts/train.py ${CONFIG_NAME} --exp_name ${EXP_NAME} ${EXTRA_ARGS[*]}"

"${TRAIN_CMD[@]}" scripts/train.py \
  "${CONFIG_NAME}" \
  --exp_name "${EXP_NAME}" \
  "${EXTRA_ARGS[@]}"
