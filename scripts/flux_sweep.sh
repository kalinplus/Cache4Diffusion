#!/usr/bin/env bash
# Native FLUX (flux/taylorseer) origin + TaylorSeer sweep.
# Configurations are dispatched in parallel across GPUS after origin finishes.
set -euo pipefail

cd "$(dirname "$0")/.."

# Keep evaluation caches on a writable, existing path on this machine.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"

GPUS="1,2,3,4"
PROMPT_FILE="${PROMPT_FILE:-assets/prompts/DrawBench200.txt}"
OUTDIR_ROOT="${OUTDIR_ROOT:-outputs}"
STEPS="${STEPS:-50}"
SEED="${SEED:-0}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"
FIRST_ENHANCE="${FIRST_ENHANCE:-3}"
BENCH_PROMPT="${BENCH_PROMPT:-a red panda wearing a top hat, photorealistic, highly detailed}"
BENCH_WARMUP="${BENCH_WARMUP:-1}"
BENCH_RUNS="${BENCH_RUNS:-1}"
WORKERS="${WORKERS:-}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"
ORIGIN_ONLY="${ORIGIN_ONLY:-0}"
CACHED_ONLY="${CACHED_ONLY:-0}"
BENCH="${BENCH:-1}"
EVAL="${EVAL:-1}"

args=(scripts/flux_sweep.py
      --gpus "$GPUS"
      --prompt_file "$PROMPT_FILE"
      --outdir_root "$OUTDIR_ROOT"
      --steps "$STEPS"
      --seed "$SEED"
      --width "$WIDTH"
      --height "$HEIGHT"
      --first_enhance "$FIRST_ENHANCE"
      --bench_prompt "$BENCH_PROMPT"
      --benchmark_warmup "$BENCH_WARMUP"
      --benchmark_runs "$BENCH_RUNS")

[ -n "${PROMPT:-}" ] && args+=(--prompt "$PROMPT")
[ -n "${MODEL_PATH:-}" ] && args+=(--model_path "$MODEL_PATH")
[ -n "${FLUX_T5_ROOT:-}" ] && args+=(--flux_t5_root "$FLUX_T5_ROOT")
[ -n "${FLUX_CLIP_ROOT:-}" ] && args+=(--flux_clip_root "$FLUX_CLIP_ROOT")
[ -n "${VARIANT:-}" ] && args+=(--variant "$VARIANT")
[ -n "$WORKERS" ] && args+=(--workers "$WORKERS")
[ "$DRY_RUN" = 1 ] && args+=(--dry-run)
[ "$RESUME" = 1 ] && args+=(--resume)
[ "$ORIGIN_ONLY" = 1 ] && args+=(--origin-only)
[ "$CACHED_ONLY" = 1 ] && args+=(--cached-only)
[ "$BENCH" = 0 ] && args+=(--no-benchmark)
[ "$EVAL" = 0 ] && args+=(--no-eval)

exec python "${args[@]}" "$@"
