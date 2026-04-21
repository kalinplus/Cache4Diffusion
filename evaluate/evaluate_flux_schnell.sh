#!/bin/bash
# Evaluate flux-schnell TaylorSeer outputs against a fixed reference folder.
# Usage: bash evaluate_flux_schnell.sh [--gpu GPU_ID]

set -euo pipefail

GPU_ID=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)   GPU_ID="$2";   shift 2 ;;
        *)       GPU_ID="$1";   shift ;;
    esac
done
export CUDA_VISIBLE_DEVICES="$GPU_ID"

ROOT="/home/hkl/Cache4Diffusion/evaluate"
PROMPT_FILE="assets/prompts/DrawBench200.txt"
TARGET_ROOT="/home/hkl/Cache4Diffusion/samples/flux-schnell/taylorseer"
REFERENCE_FOLDER="/home/hkl/Cache4Diffusion/samples/flux-schnell/taylorseer/N2O0F50Alpha0"
RESULT_FILE="$TARGET_ROOT/evaluation_results_flux_schnell.txt"

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
        echo "SKIP (no images): $label" | tee -a "$RESULT_FILE"
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

if [ ! -d "$TARGET_ROOT" ]; then
    echo "ERROR: target root not found at $TARGET_ROOT" >&2
    exit 1
fi

if [ ! -d "$REFERENCE_FOLDER" ]; then
    echo "ERROR: reference folder not found at $REFERENCE_FOLDER" >&2
    exit 1
fi

# Evaluate reference folder against itself so CLIP/ImageReward are still meaningful.
evaluate "$REFERENCE_FOLDER" "$REFERENCE_FOLDER" "$REFERENCE_FOLDER"

# Evaluate every other image-containing leaf directory against the fixed reference.
while IFS= read -r leaf_dir; do
    if [ "$leaf_dir" = "$REFERENCE_FOLDER" ]; then
        continue
    fi
    evaluate "$leaf_dir" "$REFERENCE_FOLDER" "$leaf_dir"
done < <(
    find "$TARGET_ROOT" -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) -printf '%h\n' | sort -u
)

echo "========================================"
echo "All evaluations complete. Results saved to:"
echo "  $RESULT_FILE"
