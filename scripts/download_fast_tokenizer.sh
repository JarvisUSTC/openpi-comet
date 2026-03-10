#!/usr/bin/env bash
# 在有外网的机器上预先下载 physical-intelligence/fast，供无外网集群使用。
# 用法：
#   ./scripts/download_fast_tokenizer.sh [输出目录]
#   export OPENPI_FAST_TOKENIZER_PATH=/path/to/output
#   或在 config 的 Pi0Config 中设置 fast_tokenizer_path="/path/to/output"
set -euo pipefail

OUTPUT_DIR="${1:-./fast_tokenizer}"
REPO_ID="physical-intelligence/fast"

echo "Downloading ${REPO_ID} to ${OUTPUT_DIR} ..."
mkdir -p "${OUTPUT_DIR}"

if command -v huggingface-cli &>/dev/null; then
  huggingface-cli download "${REPO_ID}" --local-dir "${OUTPUT_DIR}" --local-dir-use-symlinks False
else
  python -c "
from huggingface_hub import snapshot_download
snapshot_download('${REPO_ID}', local_dir='${OUTPUT_DIR}', local_dir_use_symlinks=False)
"
fi

echo "Done. Use: export OPENPI_FAST_TOKENIZER_PATH=$(cd "${OUTPUT_DIR}" && pwd)"
echo "Or in TrainConfig model: Pi0Config(..., fast_tokenizer_path=\"$(cd "${OUTPUT_DIR}" && pwd)\")"
