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
    | awk -v id="${TASK_ID}" 'NR>1 && $0 ~ id && index($0, "dc.com/serverless.pod-index=0") > 0 {print $1; exit}'
)"

if [[ -z "${POD}" ]]; then
  POD="$(
    bash scripts/aladdin/kubectl.sh get pods --show-labels \
      | awk -v id="${TASK_ID}" 'NR>1 && $0 ~ id {print $1; exit}'
  )"
fi

if [[ -z "${POD}" ]]; then
  # Fallback: some tasks may finish/fail before pods are discoverable (or be auto-deleted),
  # but logs are still persisted under the shared PVC.
  LOG_FILE="/capacity/vksdata/tasks/${TASK_ID}/logs/worker-0.log"
  if [[ -f "${LOG_FILE}" ]]; then
    echo "No pod found for task_id=${TASK_ID}; falling back to ${LOG_FILE}"
    tail -n "${LINES}" "${LOG_FILE}"
    exit 0
  fi
  echo "ERROR: No pod found for task_id=${TASK_ID} and no local log file at ${LOG_FILE}" >&2
  exit 1
fi

echo "Pod: ${POD}"
echo "---- /root/logs/worker-0.log (tail=${LINES}) ----"
bash scripts/aladdin/kubectl.sh exec "${POD}" -- tail -n "${LINES}" /root/logs/worker-0.log
