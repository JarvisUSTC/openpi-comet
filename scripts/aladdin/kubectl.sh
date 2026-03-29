#!/usr/bin/env bash
# Convenience wrapper for Aladdin's embedded kubectl + kubeconfig injected by the Alaya VSCode plugin.
#
# Usage:
#   bash scripts/aladdin/kubectl.sh get pods
#   bash scripts/aladdin/kubectl.sh logs <pod> --tail=200
#
# Notes:
# - Reads kubeconfig + namespace from: /root/.alaya/vscode-plugin-remote/config/config.json
# - Writes kubeconfig JSON to: /tmp/aladdin-kubeconfig.json

set -euo pipefail

PLUGIN_CFG="/root/.alaya/vscode-plugin-remote/config/config.json"
KUBECTL_BIN="/root/.alaya/vscode-plugin-remote/kubectl"
KUBECONFIG_PATH="/tmp/aladdin-kubeconfig.json"

if [[ ! -f "${PLUGIN_CFG}" ]]; then
  echo "ERROR: Missing ${PLUGIN_CFG}. Are you running inside an Alaya workshop?" >&2
  exit 1
fi
if [[ ! -x "${KUBECTL_BIN}" ]]; then
  echo "ERROR: Missing ${KUBECTL_BIN}." >&2
  exit 1
fi

NAMESPACE="$(python - <<'PY'
import json
import pathlib

cfg = json.load(open("/root/.alaya/vscode-plugin-remote/config/config.json"))
ns = cfg["debugInfo"]["namespace"]
kube_cfg = json.loads(cfg["localClusterNodes"][0]["yamlStr"])
path = pathlib.Path("/tmp/aladdin-kubeconfig.json")
path.write_text(json.dumps(kube_cfg, indent=2))
print(ns)
PY
)"

exec env KUBECONFIG="${KUBECONFIG_PATH}" "${KUBECTL_BIN}" -n "${NAMESPACE}" "$@"
