#!/usr/bin/env bash
set -euo pipefail

# Start the VLA policy server for the pick-up-from checkpoint.
# This script is intended to be run on the GPU server, for example after:
#   ssh -i ~/.ssh/id_ed25519_a800 -p 2222 root@192.168.1.182
#
# It launches through behavior-benchmark's serve_b1k_compat.py instead of
# scripts/serve_b1k.py directly. The compat launcher patches B1KPolicyWrapper
# so prompt sent by the benchmark client (obs["prompt"]) overrides the fallback
# TASK_NAME prompt.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
BENCHMARK_COMPAT="${BENCHMARK_COMPAT:-/b1k/Benchmark/benchmark/behavior-benchmark/scripts/serve_b1k_compat.py}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_b1k-knowledge_insulation-vqa-joint-skill-pick-up-from-no-task-planning}"
CKPT_DIR="${CKPT_DIR:-${OPENPI_ROOT}/outputs/checkpoints/pi05_b1k-ki-vqa-joint-pick-up-from-no-task-planning/pi05_ki_joint_pick_up_from_vqa_no_task_planning_ga1/15000}"
TASK_NAME="${TASK_NAME:-make_pizza}"
PORT="${PORT:-8000}"

if [[ -x "/b1k/hyn/openpi-codebase/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-/b1k/hyn/openpi-codebase/.venv/bin/python}"
elif [[ -x "${OPENPI_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${OPENPI_ROOT}/.venv/bin/python}"
elif [[ -x "${OPENPI_ROOT}/.venv.bak/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${OPENPI_ROOT}/.venv.bak/bin/python}"
else
  echo "No usable python found under /b1k/hyn/openpi-codebase/.venv, ${OPENPI_ROOT}/.venv, or .venv.bak" >&2
  exit 2
fi

if [[ ! -f "${BENCHMARK_COMPAT}" ]]; then
  echo "Benchmark compat launcher not found: ${BENCHMARK_COMPAT}" >&2
  exit 2
fi

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "Checkpoint directory does not exist: ${CKPT_DIR}" >&2
  exit 2
fi

if [[ ! -f "${CKPT_DIR}/_CHECKPOINT_METADATA" ]]; then
  echo "Checkpoint metadata not found: ${CKPT_DIR}/_CHECKPOINT_METADATA" >&2
  exit 2
fi

export OPENPI_ROOT
export PYTHONPATH="${OPENPI_ROOT}:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export OPENPI_FAST_TOKENIZER_PATH="${OPENPI_FAST_TOKENIZER_PATH:-/b1k/Jiawei/fast}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/b1k/.cache/openpi}"

cd "${OPENPI_ROOT}"

echo "Starting B1K VLA server"
echo "openpi_root   : ${OPENPI_ROOT}"
echo "python        : ${PYTHON_BIN}"
echo "task_name     : ${TASK_NAME}"
echo "port          : ${PORT}"
echo "policy_config : ${POLICY_CONFIG}"
echo "checkpoint    : ${CKPT_DIR}"
echo "xla_prealloc  : ${XLA_PYTHON_CLIENT_PREALLOCATE}"
echo "xla_mem_frac  : ${XLA_PYTHON_CLIENT_MEM_FRACTION}"
echo "launcher      : ${BENCHMARK_COMPAT}"
echo "prompt_source : client obs['prompt']; TASK_NAME is fallback only"
echo "fast_tok_dir  : ${OPENPI_FAST_TOKENIZER_PATH}"
echo "openpi_cache  : ${OPENPI_DATA_HOME}"
echo "host_ips      : $(hostname -I 2>/dev/null || true)"

exec "${PYTHON_BIN}" "${BENCHMARK_COMPAT}" \
  --task_name "${TASK_NAME}" \
  --port "${PORT}" \
  --control_mode receeding_horizon \
  --max_len 32 \
  --fine_grained_level 0 \
  policy:checkpoint \
  --policy.config "${POLICY_CONFIG}" \
  --policy.dir "${CKPT_DIR}"
