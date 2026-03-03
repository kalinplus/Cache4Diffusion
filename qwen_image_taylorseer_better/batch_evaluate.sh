#!/bin/bash
export CUDA_VISIBLE_DEVICES='0'

PROJECT_ROOT="/home/hkl/Cache4Diffusion"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}/taylorseer"

PROMPT_FILE="prompts/DrawBench200.txt"
BASE_DIR="samples/db200"
OUTPUT_FILE="samples/db200/evaluation_results.txt"


REFERENCE_FOLDER="/home/hkl/Cache4Diffusion/taylorseer/samples/db200/baseline"
CLIP_MODEL_PATH="/mnt/data0/pretrained_models/laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
IMAGEREWARD_MODEL_PATH="/mnt/data0/pretrained_models/zai-org/ImageReward"

# Clear previous results
> "$OUTPUT_FILE"

echo "Evaluating configurations..."
echo "================================" | tee -a "$OUTPUT_FILE"

# Function to evaluate a single directory
evaluate_dir() {
    local dir=$1
    local config_name=$(basename "$dir")

    echo "Evaluating: $config_name"

    # Write config name to output file
    echo "Config: $config_name" | tee -a "$OUTPUT_FILE"

    python evaluate.py \
        --test_folder "$dir" \
        --prompt_file "$PROMPT_FILE" \
        --reference_folder "$REFERENCE_FOLDER" \
        --clip_model_path "$CLIP_MODEL_PATH" \
        --imagereward_model_path "$IMAGEREWARD_MODEL_PATH" \
        | tee -a "$OUTPUT_FILE"

    echo "" | tee -a "$OUTPUT_FILE"
}

# Evaluate naive_ts configurations
for dir in "${BASE_DIR}/naive_ts"/*; do
    if [ -d "$dir" ]; then
        evaluate_dir "$dir"
    fi
done

# Evaluate smooth configurations
for dir in "${BASE_DIR}/smooth"/*; do
    if [ -d "$dir" ]; then
        evaluate_dir "$dir"
    fi
done

echo "================================"
echo "Evaluation completed! Results saved to: $OUTPUT_FILE"
