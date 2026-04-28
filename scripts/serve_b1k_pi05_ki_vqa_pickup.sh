#!/usr/bin/env bash
#
# Serve a pi05 + KI + VQA-joint, pick-up-from ckpt as a WebSocket VLA server.
#
# Usage:
#   ./serve_b1k_pi05_ki_vqa_pickup.sh [TASK_NAME] [STEP] [PORT]
#
# Defaults:
#   TASK_NAME = picking_up_trash
#   STEP      = 15000
#   PORT      = 8000
#
# Env overrides (optional):
#   PYTHON      python interpreter (default: /b1k/hyn/openpi-codebase/.venv/bin/python)
#   CKPT_ROOT   parent dir of step folders (so you can swap exp without editing the script)

set -euo pipefail

TASK_NAME="${1:-picking_up_trash}"
STEP="${2:-15000}"
PORT="${3:-8000}"

PYTHON="${PYTHON:-/b1k/hyn/openpi-codebase/.venv/bin/python}"
REPO_ROOT="/b1k/Jiawei/openpi-comet-clean"
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/outputs/checkpoints/pi05_b1k-ki-vqa-joint-pick-up-from-no-task-planning/pi05_ki_joint_pick_up_from_vqa_no_task_planning_ga1}"
CKPT_DIR="${CKPT_ROOT}/${STEP}"

CONFIG_NAME="pi05_b1k-knowledge_insulation-vqa-joint-skill-pick-up-from-no-task-planning"

if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "[ERR] checkpoint dir not found: ${CKPT_DIR}" >&2
    exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
    echo "[ERR] python not found / not executable: ${PYTHON}" >&2
    exit 1
fi

# L20 has no outbound internet; tell huggingface_hub / transformers to use
# local cache only and skip the HEAD-revision check that otherwise stalls
# ~30s per asset on retries.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# tokenizer.py honours OPENPI_FAST_TOKENIZER_PATH and skips HuggingFace lookup
# when set. Point it at the pre-downloaded FAST tokenizer snapshot on disk
# (downloaded via scripts/download_fast_tokenizer.sh on a machine with network).
export OPENPI_FAST_TOKENIZER_PATH="${OPENPI_FAST_TOKENIZER_PATH:-/b1k/Jiawei/fast}"

# openpi.shared.download.maybe_download() short-circuits to the local cache
# when the file already exists; without OPENPI_DATA_HOME it would default to
# /vepfs-C/.cache/openpi (mount missing on L20) and then ~/.cache/openpi
# (empty), causing a GCS fetch for paligemma_tokenizer.model. The pre-warmed
# cache lives under /b1k/.cache/openpi.
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/b1k/.cache/openpi}"

echo "==========================================================="
echo "  config       : ${CONFIG_NAME}"
echo "  ckpt         : ${CKPT_DIR}"
echo "  task_name    : ${TASK_NAME}"
echo "  port         : ${PORT}"
echo "  control_mode : receeding_horizon"
echo "  max_len      : 32   (= train action_horizon)"
echo "  python       : ${PYTHON}"
echo "  HF mode      : offline (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1)"
echo "  FAST tok dir : ${OPENPI_FAST_TOKENIZER_PATH}"
echo "  openpi cache : ${OPENPI_DATA_HOME}"
echo "==========================================================="

cd "${REPO_ROOT}"

# We launch through vqa_eval/scripts/serve_b1k_compat.py instead of
# scripts/serve_b1k.py directly, because the inference venv does not have the
# full `omnigibson` package; the compat launcher injects a minimal stub that
# wires up BEHAVIOR-1K's real network_utils.py without changing serve_b1k.py.
exec "${PYTHON}" "${REPO_ROOT}/vqa_eval/scripts/serve_b1k_compat.py" \
    --port "${PORT}" \
    --task_name "${TASK_NAME}" \
    --control_mode receeding_horizon \
    --max_len 32 \
    --fine_grained_level 0 \
    policy:checkpoint \
    --policy.config "${CONFIG_NAME}" \
    --policy.dir "${CKPT_DIR}"
