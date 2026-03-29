#!/usr/bin/env bash
# Watch logs for an Aladdin "task" by task id (dc.com/serverless.biz-id).
#
# Usage:
#   bash scripts/aladdin/watch_task_logs.sh <task_id> [--lines N]
#
# Example:
#   bash scripts/aladdin/watch_task_logs.sh 660a5a76-... --lines 200

set -euo pipefail

TASK_ID="${1:?missing <task_id> (dc.com/serverless.biz-id)}"
shift 1

LINES=200
if [[ "${1:-}" == "--lines" ]]; then
  LINES="${2:?missing N}"
  shift 2
fi

POD="$(
  bash scripts/aladdin/kubectl.sh get pods --show-labels \
    | awk -v id="${TASK_ID}" 'NR>1 && $0 ~ id {print $1; exit}'
)"

if [[ -z "${POD}" ]]; then
  echo "ERROR: No pod found for task_id=${TASK_ID}" >&2
  exit 1
fi

echo "Pod: ${POD}"
echo "---- /root/logs/worker-0.log (tail=${LINES}) ----"
bash scripts/aladdin/kubectl.sh exec "${POD}" -- tail -n "${LINES}" /root/logs/worker-0.log

