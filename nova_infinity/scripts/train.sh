#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

exec "${TORCHRUN:-torchrun}" \
  --nproc_per_node "${NPROC_PER_NODE:-1}" \
  --nnodes "${NNODES:-1}" \
  --node_rank "${NODE_RANK:-0}" \
  --master_addr "${MASTER_ADDR:-127.0.0.1}" \
  --master_port "${MASTER_PORT:-29500}" \
  train.py "$@"
