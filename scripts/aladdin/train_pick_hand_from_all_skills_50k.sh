#!/usr/bin/env bash
# Aladdin task entrypoint for pick_hand experiments initialized from all_skills 50k.
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

CONFIG_NAME="${CONFIG_NAME:?missing CONFIG_NAME}"
EXP_NAME="${EXP_NAME:?missing EXP_NAME}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/aladdin_logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXP_NAME}.log}"

echo "Config:   ${CONFIG_NAME}"
echo "Exp name: ${EXP_NAME}"
echo "Log file: ${LOG_FILE}"

set -o pipefail
bash "${REPO_ROOT}/scripts/train_multinode_jax.sh" "${CONFIG_NAME}" "${EXP_NAME}" 2>&1 | tee -a "${LOG_FILE}"
