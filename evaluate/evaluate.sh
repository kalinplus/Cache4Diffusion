#!/bin/bash
# Evaluate flux_diffusers nf4 output directories against a fixed reference.
# Usage: bash evaluate.sh [--gpu GPU_ID]

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
TARGET_ROOT="/home/hkl/Cache4Diffusion/flux_diffusers/outputs/float16/lora-none/quant-nf4"
RESULT_FILE="$TARGET_ROOT/evaluation_results.txt"

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

# ------------------------------------------------------------------ #
# New evaluation target: flux_diffusers nf4 outputs
# reference_folder: S50/N0O0F50A0
# support both layouts under quant-nf4:
#   1) nested: S10|S16|S50/N... (e.g. S10/N0O0F50A0)
#   2) direct: N...             (e.g. N3O0F3A0)
# ------------------------------------------------------------------ #
REFERENCE_FOLDER="$TARGET_ROOT/S50/N0O0F50A0"

if [ ! -d "$REFERENCE_FOLDER" ]; then
    echo "ERROR: reference folder not found at $REFERENCE_FOLDER" >&2
    exit 1
fi

# Evaluate reference folder against itself so CLIP/ImageReward are still reported.
evaluate "$REFERENCE_FOLDER" "$REFERENCE_FOLDER" "$REFERENCE_FOLDER"

while IFS= read -r leaf_dir; do
    if [ "$leaf_dir" = "$REFERENCE_FOLDER" ]; then
        continue
    fi
    label="$leaf_dir"
    evaluate "$leaf_dir" "$REFERENCE_FOLDER" "$label"
done < <(
    find "$TARGET_ROOT" -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) -printf '%h\n' | sort -u
)

echo "========================================"
echo "All evaluations complete. Results saved to:"
echo "  $RESULT_FILE"
