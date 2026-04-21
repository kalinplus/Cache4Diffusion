#!/bin/bash
# Legacy evaluator for original hunyuan/flux output layouts.
# Usage: bash evaluate_legacy.sh [--gpu GPU_ID] [--model MODEL]
#
# Models:
#   hunyuan (default)
#     outputs/origin/{with,without}_refiner/    - reference baselines
#     outputs/naive_ts/{with,without}_refiner/  - TaylorSeer without smoothing
#     outputs/smooth/exp/**                     - smoothing experiments
#
#   flux
#     outputs/flux/baseline/                    - reference baseline
#     outputs/flux/naive_ts/interval*/          - TaylorSeer without smoothing
#     outputs/flux/smooth/{alpha}/interval*/    - smoothing experiments

set -euo pipefail

GPU_ID=0
MODEL="hunyuan"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)   GPU_ID="$2";   shift 2 ;;
        --model) MODEL="$2";    shift 2 ;;
        *)       GPU_ID="$1";   shift ;;
    esac
done
export CUDA_VISIBLE_DEVICES="$GPU_ID"

ROOT="/home/hkl/Cache4Diffusion/evaluate"
OUTPUTS="outputs"
PROMPT_FILE="assets/prompts/DrawBench200.txt"
RESULT_FILE="$OUTPUTS/evaluation_results_${MODEL}.txt"

# Clear previous results
> "$RESULT_FILE"

evaluate() {
    local test_folder="$1"
    local ref_folder="$2"
    local label="$3"

    # Skip if test folder has no images
    local img_count
    img_count=$(find "$test_folder" -maxdepth 1 \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) -print -quit)
    if [ -z "$img_count" ]; then
        echo "SKIP (no images): $label"
        return
    fi

    echo "========================================" | tee -a "$RESULT_FILE"
    echo "Evaluating: $label" | tee -a "$RESULT_FILE"
    echo "  test:  $test_folder" | tee -a "$RESULT_FILE"
    echo "  ref:   $ref_folder" | tee -a "$RESULT_FILE"
    echo "========================================" | tee -a "$RESULT_FILE"

    python "$ROOT/evaluate.py" \
        --test_folder "$test_folder" \
        --prompt_file "$PROMPT_FILE" \
        --reference_folder "$ref_folder" \
        2>&1 | tee -a "$RESULT_FILE" || echo "ERROR: evaluation failed for $label" | tee -a "$RESULT_FILE"

    echo "" | tee -a "$RESULT_FILE"
}

if [ "$MODEL" = "flux" ]; then
    FLUX_OUT="$OUTPUTS/flux"
    REF="$FLUX_OUT/baseline"

    if [ ! -d "$REF" ]; then
        echo "ERROR: flux baseline not found at $REF" >&2
        exit 1
    fi

    # baseline self-eval for absolute metrics (CLIP, ImageReward)
    evaluate "$REF" "$REF" "flux/baseline"

    # naive_ts - one dir per interval/order combination
    if [ -d "$FLUX_OUT/naive_ts" ]; then
        while IFS= read -r leaf_dir; do
            label="flux/${leaf_dir#$FLUX_OUT/}"
            evaluate "$leaf_dir" "$REF" "$label"
        done < <(
            find "$FLUX_OUT/naive_ts" -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) -printf '%h\n' | sort -u
        )
    fi

    # smooth - auto-discover all leaf dirs
    if [ -d "$FLUX_OUT/smooth" ]; then
        while IFS= read -r leaf_dir; do
            label="flux/${leaf_dir#$FLUX_OUT/}"
            evaluate "$leaf_dir" "$REF" "$label"
        done < <(
            find "$FLUX_OUT/smooth" -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) -printf '%h\n' | sort -u
        )
    fi

else
    REF_WITH="$OUTPUTS/origin/with_refiner"
    REF_WITHOUT="$OUTPUTS/origin/without_refiner"

    # baseline self-eval for absolute metrics (CLIP, ImageReward)
    [ -d "$REF_WITH" ]    && evaluate "$REF_WITH"    "$REF_WITH"    "origin/with_refiner"
    [ -d "$REF_WITHOUT" ] && evaluate "$REF_WITHOUT" "$REF_WITHOUT" "origin/without_refiner"

    # naive_ts
    for variant in with_refiner without_refiner; do
        test_dir="$OUTPUTS/naive_ts/$variant"
        if [ "$variant" = "with_refiner" ]; then
            ref_dir="$REF_WITH"
        else
            ref_dir="$REF_WITHOUT"
        fi
        [ -d "$test_dir" ] && evaluate "$test_dir" "$ref_dir" "naive_ts/$variant"
    done

    # smooth/exp - auto-discover leaf dirs containing images
    if [ -d "$OUTPUTS/smooth/exp" ]; then
        while IFS= read -r leaf_dir; do
            label="${leaf_dir#$OUTPUTS/}"
            evaluate "$leaf_dir" "$REF_WITHOUT" "$label"
        done < <(
            find "$OUTPUTS/smooth/exp" -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) -printf '%h\n' | sort -u
        )
    fi
fi

echo "========================================"
echo "All evaluations complete. Results saved to:"
echo "  $RESULT_FILE"
