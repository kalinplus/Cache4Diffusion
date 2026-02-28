#!/bin/bash
# Evaluate all output directories against origin references.
# Usage: bash run_eval.sh [--gpu GPU_ID]
#
# Directory structure:
#   outputs/origin/{with,without}_refiner/  — reference baselines
#   outputs/naive_ts/{with,without}_refiner/ — TaylorSeer without smoothing
#   outputs/smooth/exp/{method}/{config}/    — smoothing experiments
#
# Reference matching:
#   - naive_ts/with_refiner    → origin/with_refiner
#   - naive_ts/without_refiner → origin/without_refiner
#   - smooth/exp/**            → origin/without_refiner

set -euo pipefail

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUTPUTS="$ROOT/outputs"
PROMPT_FILE="$ROOT/assets/prompts/DrawBench200.txt"
RESULT_FILE="$OUTPUTS/evaluation_results.txt"

REF_WITH="$OUTPUTS/origin/with_refiner"
REF_WITHOUT="$OUTPUTS/origin/without_refiner"

# Clear previous results
> "$RESULT_FILE"

run_eval() {
    local test_folder="$1"
    local ref_folder="$2"
    local label="$3"

    # Skip if test folder has no images
    local img_count
    img_count=$(find "$test_folder" -maxdepth 1 -name '*.png' -o -name '*.jpg' | head -1)
    if [ -z "$img_count" ]; then
        echo "SKIP (no images): $label"
        return
    fi

    echo "========================================" | tee -a "$RESULT_FILE"
    echo "Evaluating: $label" | tee -a "$RESULT_FILE"
    echo "  test:  $test_folder" | tee -a "$RESULT_FILE"
    echo "  ref:   $ref_folder" | tee -a "$RESULT_FILE"
    echo "========================================" | tee -a "$RESULT_FILE"

    conda run -n eval python "$ROOT/evaluate.py" \
        --test_folder "$test_folder" \
        --prompt_file "$PROMPT_FILE" \
        --reference_folder "$ref_folder" \
        2>&1 | tee -a "$RESULT_FILE"

    echo "" | tee -a "$RESULT_FILE"
}

# --- 1. naive_ts ---
for variant in with_refiner without_refiner; do
    test_dir="$OUTPUTS/naive_ts/$variant"
    if [ "$variant" = "with_refiner" ]; then
        ref_dir="$REF_WITH"
    else
        ref_dir="$REF_WITHOUT"
    fi
    [ -d "$test_dir" ] && run_eval "$test_dir" "$ref_dir" "naive_ts/$variant"
done

# --- 2. smooth/exp — auto-discover leaf dirs containing .png files ---
if [ -d "$OUTPUTS/smooth/exp" ]; then
    # Find all directories that directly contain .png files (leaf image dirs)
    while IFS= read -r leaf_dir; do
        # Build a human-readable label from the relative path
        label="${leaf_dir#$OUTPUTS/}"
        run_eval "$leaf_dir" "$REF_WITHOUT" "$label"
    done < <(
        find "$OUTPUTS/smooth/exp" -name '*.png' -printf '%h\n' | sort -u
    )
fi

echo "========================================"
echo "All evaluations complete. Results saved to:"
echo "  $RESULT_FILE"
