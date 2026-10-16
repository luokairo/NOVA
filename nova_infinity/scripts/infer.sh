#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
weights_dir="${WEIGHTS_DIR:-weights}"

exec "${PYTHON:-python3}" -m tools.run_infinity \
  --pn "${PN:-1M}" \
  --model_type "${MODEL_TYPE:-infinity_2b}" \
  --model_path "${MODEL_PATH:-$weights_dir/infinity_2b_reg.pth}" \
  --vae_path "${VAE_PATH:-$weights_dir/infinity_vae_d32reg.pth}" \
  --text_encoder_ckpt "${TEXT_ENCODER_CKPT:-$weights_dir/flan-t5-xl}" \
  --checkpoint_type "${CHECKPOINT_TYPE:-torch}" \
  --vae_type "${VAE_TYPE:-32}" \
  --apply_spatial_patchify "${APPLY_SPATIAL_PATCHIFY:-0}" \
  --cfg "${CFG:-4}" --tau "${TAU:-0.5}" \
  --save_file "${SAVE_FILE:-outputs/sample.png}" \
  "$@"
