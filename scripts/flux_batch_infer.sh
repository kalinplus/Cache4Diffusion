#!/usr/bin/env bash
# Parallel batch inference: 19 experiments across 6 GPUs in 4 rounds.
# Usage: MODEL_NAME=qwen_image bash batch_infer.sh

MODEL_NAME="${MODEL_NAME:-flux}"

if [ "$MODEL_NAME" = "flux" ]; then
    model_path="/mnt/data0/pretrained_models/black-forest-labs/FLUX.1-dev"
    dtype="float16"
    extra_args=""
elif [ "$MODEL_NAME" = "qwen_image" ]; then
    model_path="/mnt/data0/pretrained_models/Qwen/Qwen-Image"
    dtype="bfloat16"
    extra_args=""
elif [ "$MODEL_NAME" = "hunyuan_image" ]; then
    model_path="/mnt/data0/pretrained_models/tencent/HunyuanImage-2.1"
    dtype="bfloat16"
    extra_args=""
else
    echo "Unknown MODEL_NAME: $MODEL_NAME (supported: flux, qwen_image, hunyuan_image)"
    exit 1
fi

PYTHON="/home/hkl/miniconda3/envs/qwenimage/bin/python"
export PYTHONPATH="/home/hkl/Cache4Diffusion:${PYTHONPATH:-}"
export DIFFUSERS_ATTN_BACKEND=flash

prompt_file="assets/prompts/DrawBench200.txt"
base_outdir="outputs/${MODEL_NAME}"

GPUS=(0 1 2 3 6 7)

# ── Helper: launch one experiment in the background ────────────────────────────
# Usage: launch GPU OUTDIR STRATEGY [INTERVAL ORDER USE_SMOOTH ALPHA]
# Logs go to $OUTDIR/infer.log. After calling, use $! to capture PID.
launch() {
    local gpu="$1" outdir="$2" strategy="$3"
    local interval="${4:-6}" order="${5:-1}" use_smooth="${6:-False}" alpha="${7:-0.8}"

    mkdir -p "$outdir"
    echo "[GPU $gpu] START  strategy=$strategy interval=$interval order=$order smooth=$use_smooth alpha=$alpha"
    echo "          -> $outdir/infer.log"

    CUDA_VISIBLE_DEVICES="$gpu" \
    USE_HYBRID_SMOOTHING="False" \
    SMOOTHING_METHOD="exponential" \
    USE_SMOOTHING="$use_smooth" \
    SMOOTHING_ALPHA="$alpha" \
    TS_CACHE_INTERVAL="$interval" \
    TS_MAX_ORDER="$order" \
    "$PYTHON" batch_infer.py \
        --model        "$model_path" \
        --model_name   "$MODEL_NAME" \
        --strategy     "$strategy" \
        --steps        50 \
        --seed         42 \
        --dtype        "$dtype" \
        --guidance_scale 7.5 \
        --outdir       "$outdir" \
        --prefix       "${MODEL_NAME}" \
        --prompt_file  "$prompt_file" \
        $extra_args \
        > "$outdir/infer.log" 2>&1 &
}

wait_round() {
    local label="$1"; shift
    echo ""
    echo "═══ Waiting: $label ═══"
    local any_err=0
    for pid in "$@"; do
        if ! wait "$pid"; then
            echo "[WARN] PID $pid exited with error"
            any_err=1
        fi
    done
    echo "═══ Done: $label (errors=$any_err) ═══"
    echo ""
}

# ── Round 1: naive_ts — 6 runs, USE_SMOOTHING=False ───────────────────────────
pids=()
i=0
for interval in 4 5 6; do
    for order in 1 2; do
        outdir="${base_outdir}/naive_ts/interval${interval}_order${order}"
        launch "${GPUS[$i]}" "$outdir" taylorseer "$interval" "$order" False
        pids+=("$!")
        ((i++))
    done
done
wait_round "Round 1: naive_ts (6 runs)" "${pids[@]}"

# ── Build smooth run list: interval × order × alpha → 12 entries ───────────────
smooth_args=()
for interval in 4 5 6; do
    for order in 1 2; do
        for alpha in 0.8 0.9; do
            smooth_args+=("$interval|$order|$alpha")
        done
    done
done

# ── Round 2: smooth — runs 0-5 (all interval×order with alpha=0.8) ─────────────
pids=()
for i in 0 1 2 3 4 5; do
    IFS='|' read -r interval order alpha <<< "${smooth_args[$i]}"
    outdir="${base_outdir}/smooth/${alpha}/interval${interval}_order${order}"
    launch "${GPUS[$i]}" "$outdir" taylorseer "$interval" "$order" True "$alpha"
    pids+=("$!")
done
wait_round "Round 2: smooth batch 1 (runs 0-5)" "${pids[@]}"

# ── Round 3: smooth — runs 6-11 (all interval×order with alpha=0.9) ───────────
pids=()
i=0
for j in 6 7 8 9 10 11; do
    IFS='|' read -r interval order alpha <<< "${smooth_args[$j]}"
    outdir="${base_outdir}/smooth/${alpha}/interval${interval}_order${order}"
    launch "${GPUS[$i]}" "$outdir" taylorseer "$interval" "$order" True "$alpha"
    pids+=("$!")
    ((i++))
done
wait_round "Round 3: smooth batch 2 (runs 6-11)" "${pids[@]}"

# ── Round 4: baseline — 1 run, no caching ─────────────────────────────────────
outdir="${base_outdir}/baseline"
launch "${GPUS[0]}" "$outdir" none
wait_round "Round 4: baseline" "$!"

echo "All experiments complete. Results in: ${base_outdir}/"
echo ""
echo "Output layout:"
echo "  naive_ts/interval{N}_order{N}/   — 6 dirs"
echo "  smooth/{alpha}/interval{N}_order{N}/  — 12 dirs (alpha: 0.8 0.9)"
echo "  baseline/                         — 1 dir"
