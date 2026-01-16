#!/bin/bash
export CUDA_VISIBLE_DEVICES='0'

PROJECT_ROOT="/home/hkl/Cache4Diffusion"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

PROMPT_FILE="assets/prompts/DrawBench200.txt"
OUTPUTS_ROOT="/home/hkl/Cache4Diffusion/outputs"
OUTPUT_FILE="${OUTPUTS_ROOT}/evaluation_results.txt"

REFERENCE_FOLDER="${OUTPUTS_ROOT}/origin/with_refiner"
CLIP_MODEL_PATH="/mnt/data0/pretrained_models/laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
IMAGEREWARD_MODEL_PATH="/mnt/data0/pretrained_models/zai-org/ImageReward"

# Clear previous results
> "$OUTPUT_FILE"

echo "Evaluating configurations..."
echo "================================" | tee -a "$OUTPUT_FILE"

# Function to evaluate a single directory
evaluate_dir() {
    local dir=$1
    local config_name=$2

    if [ ! -d "$dir" ]; then
        echo "Skipping: $config_name (directory not found: $dir)" | tee -a "$OUTPUT_FILE"
        return
    fi

    # Check if directory has images
    if [ -z "$(ls -A "$dir" 2>/dev/null)" ]; then
        echo "Skipping: $config_name (empty directory)" | tee -a "$OUTPUT_FILE"
        return
    fi

    echo "Evaluating: $config_name"
    echo "Config: $config_name" | tee -a "$OUTPUT_FILE"
    echo "Path: $dir" | tee -a "$OUTPUT_FILE"

    python HunyuanImage-2.1/evaluate.py \
        --test_folder "$dir" \
        --prompt_file "$PROMPT_FILE" \
        --reference_folder "$REFERENCE_FOLDER" \
        --clip_model_path "$CLIP_MODEL_PATH" \
        --imagereward_model_path "$IMAGEREWARD_MODEL_PATH" \
        | tee -a "$OUTPUT_FILE"

    echo "" | tee -a "$OUTPUT_FILE"
}

echo "================================" | tee -a "$OUTPUT_FILE"
echo "0. Evaluating Origin HunyuanImage" | tee -a "$OUTPUT_FILE"
echo "================================" | tee -a "$OUTPUT_FILE"

evaluate_dir "${OUTPUTS_ROOT}/origin/with_refiner" "origin_with_refiner"
evaluate_dir "${OUTPUTS_ROOT}/origin/without_refiner" "origin_without_refiner"

echo "================================" | tee -a "$OUTPUT_FILE"
echo "1. Evaluating Naive TaylorSeer (no smoothing)" | tee -a "$OUTPUT_FILE"
echo "================================" | tee -a "$OUTPUT_FILE"

# Evaluate naive_ts configurations
evaluate_dir "${OUTPUTS_ROOT}/naive_ts/with_refiner" "naive_ts_with_refiner"
evaluate_dir "${OUTPUTS_ROOT}/naive_ts/without_refiner" "naive_ts_without_refiner"

echo "================================" | tee -a "$OUTPUT_FILE"
echo "2. Evaluating Exponential Smoothing" | tee -a "$OUTPUT_FILE"
echo "================================" | tee -a "$OUTPUT_FILE"

# Evaluate exponential smoothing configurations
for alpha in 0.75 0.8 0.85 0.9 0.95; do
    evaluate_dir "${OUTPUTS_ROOT}/smooth/exp/exponential/${alpha}" "exponential_alpha_${alpha}"
done

echo "================================" | tee -a "$OUTPUT_FILE"
echo "3. Evaluating Moving Average Smoothing" | tee -a "$OUTPUT_FILE"
echo "================================" | tee -a "$OUTPUT_FILE"

# Evaluate moving average smoothing
evaluate_dir "${OUTPUTS_ROOT}/smooth/exp/moving_average/0.8" "moving_average_alpha_0.8"

echo "================================"
echo "Evaluation completed! Results saved to: $OUTPUT_FILE"
echo "================================"
