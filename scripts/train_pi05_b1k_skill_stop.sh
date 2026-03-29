#!/usr/bin/env bash
set -euo pipefail

# JAX memory behavior (tune as needed for your machine)
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

config_name="${1:-pi05_b1k-skill_stop}"
exp_name="${2:-openpi_$(date +%Y%m%d_%H%M%S)}"

uv run scripts/train.py \
  "${config_name}" \
  --exp_name="${exp_name}"

