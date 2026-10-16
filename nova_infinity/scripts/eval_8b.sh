#!/usr/bin/env bash
set -euo pipefail

weights_dir="${WEIGHTS_DIR:-weights}"
export TEXT_ENCODER_CKPT="${TEXT_ENCODER_CKPT:-$weights_dir/flan-t5-xl}"
export MODEL_TYPE="${MODEL_TYPE:-infinity_8b}"
export MODEL_PATH="${MODEL_PATH:-$weights_dir/infinity_8b_weights}"
export VAE_PATH="${VAE_PATH:-$weights_dir/infinity_vae_d56_f8_14_patchify.pth}"
export CHECKPOINT_TYPE="${CHECKPOINT_TYPE:-torch_shard}"
export VAE_TYPE="${VAE_TYPE:-14}"
export APPLY_SPATIAL_PATCHIFY="${APPLY_SPATIAL_PATCHIFY:-1}"
export CFG="${CFG:-3}"
benchmark="${1:?Usage: bash scripts/eval_8b.sh <geneval|image_reward|mjhq|hpsv2> [arguments]}"
shift
if [[ -n "${NOVA_ENTROPY_THRESHOLD:-}" ]]; then
  set -- --nova_entropy_threshold "$NOVA_ENTROPY_THRESHOLD" "$@"
fi
exec bash "$(dirname "${BASH_SOURCE[0]}")/eval.sh" "$benchmark" "$@"
