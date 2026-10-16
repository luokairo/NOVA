# Evaluation

| Benchmark | Generate images | Score images |
| --- | --- | --- |
| GenEval | `bash scripts/eval.sh geneval [arguments]` | `evaluation/gen_eval/evaluate_images.py` |
| ImageReward | `bash scripts/eval.sh image_reward [arguments]` | `evaluation/image_reward/cal_imagereward.py` |
| MJHQ | `bash scripts/eval.sh mjhq [arguments]` | `tools/fid_score.py` |
| HPSv2 | `bash scripts/eval.sh hpsv2 [arguments]` | HPSv2 scoring tools |

Run the generation commands from `nova_infinity/`. Supply your checkpoint,
text encoder, benchmark data, and output paths through arguments or environment
variables. The 8B wrapper is `scripts/eval_8b.sh`.

For GenEval, use `--model-config` and `--model-path` for the object detector;
`--options clip_weights=<checkpoint>` selects local CLIP weights. ImageReward
requires `--download_root` to specify its model directory.
