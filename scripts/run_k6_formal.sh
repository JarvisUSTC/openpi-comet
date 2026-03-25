#!/bin/bash
# K6 formal training: 50k steps, video memory K=6
# Usage: bash scripts/run_k6_formal.sh

set -euo pipefail

# Activate virtual environment if available
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

CONFIG_NAME="pi05_b1k-k6-smoke-step50000"
LOG_DIR="./outputs/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/${CONFIG_NAME}_${TIMESTAMP}.log"

echo "=========================================="
echo " K=6 Formal Training"
echo " Config: ${CONFIG_NAME}"
echo " Log:    ${LOG_FILE}"
echo "=========================================="

python scripts/train.py "$CONFIG_NAME" 2>&1 | tee "$LOG_FILE"
