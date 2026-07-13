#!/usr/bin/env bash
#
# run.sh — single entry point to run ANY Cache4Diffusion model.
#
# It is a thin wrapper around run.py (the dispatcher).  You can either pass
# flags directly, or set a few convenience environment variables:
#
#   MODEL     -> --model <name>      (see: bash run.sh --list)
#   GPU       -> --gpu <ids>         (CUDA_VISIBLE_DEVICES)
#   MODE      -> --mode single|batch
#   CONDA_ENV -> --conda_env <name>  (override the per-model env)
#   PYTHON    -> --python <path>     (skip `conda activate`, use this interpreter)
#
# ── Examples ──────────────────────────────────────────────────────────────
#   bash run.sh --list
#   bash run.sh --list --task video_gen
#
#   # image generation
#   MODEL=flux_diffusers GPU=0 bash run.sh --prompt "a cat" --steps 50
#   bash run.sh --model qwen_image --prompt "a cat" --steps 50 --dry_run
#   # unified output path + auto-eval after generation:
#   bash run.sh --model flux_diffusers --prompt "a cat" --steps 50 --eval --gpu 0
#   bash run.sh --model hunyuan_image --prompt_file assets/prompts/DrawBench200.txt
#
#   # image editing
#   bash run.sh --model flux_kontext --prompt "make it night" --input_image img.jpg
#   bash run.sh --model qwen_edit --dataset_path /path/to/GEdit-Bench
#
#   # video generation
#   bash run.sh --model hunyuan_video --prompt "a cat walks on grass" \
#       --video_length 65 --video_size 544 960
#
#   # dry-run any of the above to see the exact command without launching
#   bash run.sh --model hunyuan_video --prompt "..." --dry_run
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Interpreter that runs the dispatcher itself (stdlib only — any python works).
DISPATCH_PY="${PYTHON:-python3}"

# Translate convenience env vars into dispatcher flags (only when set).
passthrough=()
[ -n "${MODEL:-}" ]      && passthrough+=( --model "$MODEL" )
[ -n "${GPU:-}" ]        && passthrough+=( --gpu "$GPU" )
[ -n "${MODE:-}" ]       && passthrough+=( --mode "$MODE" )
[ -n "${CONDA_ENV:-}" ]  && passthrough+=( --conda_env "$CONDA_ENV" )
[ -n "${NPROC:-}" ]      && passthrough+=( --nproc "$NPROC" )

# If PYTHON was set, also forward it so the launched job uses that interpreter
# (skips conda activation).  When not set, the dispatcher picks the conda env.
if [ -n "${PYTHON:-}" ]; then
    passthrough+=( --python "$PYTHON" )
    # the dispatcher itself must run on *some* python though:
    DISPATCH_PY="python3"
fi

exec "$DISPATCH_PY" run.py "${passthrough[@]}" "$@"
