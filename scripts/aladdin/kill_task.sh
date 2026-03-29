#!/usr/bin/env bash
# Kill an Aladdin "task" by deleting its pod(s) (best-effort).
#
# Usage:
#   bash scripts/aladdin/kill_task.sh <task_id>

set -euo pipefail

TASK_ID="${1:?missing <task_id> (dc.com/serverless.biz-id)}"

PODS="$(
  bash scripts/aladdin/kubectl.sh get pods --show-labels \
    | awk -v id="${TASK_ID}" 'NR>1 && $0 ~ id {print $1}'
)"

if [[ -z "${PODS}" ]]; then
  echo "No pods found for task_id=${TASK_ID} (already finished?)"
  exit 0
fi

echo "Deleting pods:"
echo "${PODS}"

for p in ${PODS}; do
  bash scripts/aladdin/kubectl.sh delete pod "${p}" --wait=false || true
done

