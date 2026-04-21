#!/bin/bash
# Evaluate flux_simple float16 LoRA outputs against per-LoRA fixed reference folders.
# Usage: bash evaluate_flux_simple_float16_lora.sh [--gpu GPU_ID]

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
PROMPT_FILE="/home/hkl/Cache4Diffusion/assets/prompts/DrawBench200.txt"
FLOAT16_ROOT="/home/hkl/Cache4Diffusion/flux_simple/outputs/float16"

evaluate_one_folder() {
    local test_folder="$1"
    local ref_folder="$2"
    local label="$3"
    local result_file="$4"

    # Skip if test folder has no images.
    local img_any
    img_any=$(find "$test_folder" -maxdepth 1 \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) -print -quit)
    if [ -z "$img_any" ]; then
        echo "SKIP (no images): $label" | tee -a "$result_file"
        return
    fi

    echo "========================================" | tee -a "$result_file"
    echo "Evaluating: $label" | tee -a "$result_file"
    echo "  test:  $test_folder" | tee -a "$result_file"
    echo "  ref:   $ref_folder" | tee -a "$result_file"
    echo "========================================" | tee -a "$result_file"

    python "$ROOT/evaluate.py" \
        --test_folder "$test_folder" \
        --prompt_file "$PROMPT_FILE" \
        --reference_folder "$ref_folder" \
        2>&1 | tee -a "$result_file" || echo "ERROR: evaluation failed for $label" | tee -a "$result_file"

    echo "" | tee -a "$result_file"
}

if [ ! -d "$FLOAT16_ROOT" ]; then
    echo "ERROR: float16 root not found at $FLOAT16_ROOT" >&2
    exit 1
fi

if [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: prompt file not found at $PROMPT_FILE" >&2
    exit 1
fi

while IFS= read -r lora_dir; do
    lora_name=$(basename "$lora_dir")
    reference_folder="$lora_dir/quant-none/S50/N0O0F50A0"
    result_file="$lora_dir/evaluation_results_float16.txt"

    if [ ! -d "$reference_folder" ]; then
        echo "WARN: reference folder missing for $lora_name, skip: $reference_folder"
        continue
    fi

    # Clear previous result file for each LoRA.
    > "$result_file"

    echo "========================================" | tee -a "$result_file"
    echo "LoRA: $lora_name" | tee -a "$result_file"
    echo "Result file: $result_file" | tee -a "$result_file"
    echo "Reference folder: $reference_folder" | tee -a "$result_file"
    echo "========================================" | tee -a "$result_file"
    echo "" | tee -a "$result_file"

    # Evaluate reference folder against itself first.
    evaluate_one_folder "$reference_folder" "$reference_folder" "$reference_folder" "$result_file"

    # Evaluate all other image-containing directories recursively under this LoRA directory.
    while IFS= read -r leaf_dir; do
        if [ "$leaf_dir" = "$reference_folder" ]; then
            continue
        fi
        evaluate_one_folder "$leaf_dir" "$reference_folder" "$leaf_dir" "$result_file"
    done < <(
        find "$lora_dir" -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) -printf '%h\n' | sort -u
    )

    echo "========================================" | tee -a "$result_file"
    echo "LoRA complete: $lora_name" | tee -a "$result_file"
    echo "Results saved to: $result_file" | tee -a "$result_file"
    echo "========================================" | tee -a "$result_file"
    echo ""
done < <(
    find "$FLOAT16_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'lora-*' ! -name 'lora-none' | sort
)

echo "========================================"
echo "All requested LoRA evaluations complete."
echo "Per-LoRA result files are saved under each LoRA directory."