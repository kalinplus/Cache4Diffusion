#!/bin/bash
export CUDA_VISIBLE_DEVICES='0'

PROJECT_ROOT="/home/hkl/Cache4Diffusion"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

PROMPT_FILE="assets/prompts/DrawBench200.txt"
OUTPUTS_ROOT="/home/hkl/Cache4Diffusion/outputs"
OUTPUT_FILE="${OUTPUTS_ROOT}/evaluation_results.txt"

REFERENCE_FOLDER="${OUTPUTS_ROOT}/origin/without_refiner"
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
echo "2. Evaluating Exponential Smoothing with Cache Parameters" | tee -a "$OUTPUT_FILE"
echo "================================" | tee -a "$OUTPUT_FILE"

# Evaluate exponential smoothing with different cache parameters
# Directory structure: smooth/exp/exponential/N{interval}O{max_order}F{first_enhance}/{alpha}
# Also supports old format: smooth/exp/exponential/{alpha} (without cache params)
EXP_ROOT="${OUTPUTS_ROOT}/smooth/exp/exponential"

if [ -d "$EXP_ROOT" ]; then
    # First, check for new format with N*O*F cache param directories
    has_cache_dirs=false
    for cache_dir in "$EXP_ROOT"/N*O*F*; do
        if [ ! -d "$cache_dir" ]; then
            continue
        fi
        has_cache_dirs=true

        # Extract cache params from directory name (e.g., "N5O2F3")
        cache_params=$(basename "$cache_dir")

        # Iterate over alpha subdirectories
        for alpha_dir in "$cache_dir"/*; do
            if [ ! -d "$alpha_dir" ]; then
                continue
            fi

            alpha=$(basename "$alpha_dir")
            config_name="exponential_${cache_params}_alpha_${alpha}"

            echo "----------------------------------------" | tee -a "$OUTPUT_FILE"
            echo "Evaluating: $config_name" | tee -a "$OUTPUT_FILE"
            echo "Cache params: $cache_params" | tee -a "$OUTPUT_FILE"
            echo "Alpha: $alpha" | tee -a "$OUTPUT_FILE"
            echo "Path: $alpha_dir" | tee -a "$OUTPUT_FILE"

            evaluate_dir "$alpha_dir" "$config_name"
        done
    done

    # If no cache param directories found, check for old format (direct alpha directories)
    if [ "$has_cache_dirs" = false ]; then
        echo "No N*O*F directories found, checking old format (direct alpha directories)..." | tee -a "$OUTPUT_FILE"
        for alpha_dir in "$EXP_ROOT"/*; do
            if [ ! -d "$alpha_dir" ]; then
                continue
            fi

            alpha=$(basename "$alpha_dir")
            # Skip if it's a cache param directory
            case "$alpha" in
                N*O*F*) continue ;;
            esac

            config_name="exponential_alpha_${alpha}"

            echo "----------------------------------------" | tee -a "$OUTPUT_FILE"
            echo "Evaluating: $config_name (old format)" | tee -a "$OUTPUT_FILE"
            echo "Alpha: $alpha" | tee -a "$OUTPUT_FILE"
            echo "Path: $alpha_dir" | tee -a "$OUTPUT_FILE"

            evaluate_dir "$alpha_dir" "$config_name"
        done
    fi
else
    echo "Exponential smoothing directory not found: $EXP_ROOT" | tee -a "$OUTPUT_FILE"
fi

echo "================================" | tee -a "$OUTPUT_FILE"
echo "3. Evaluating Moving Average Smoothing with Cache Parameters" | tee -a "$OUTPUT_FILE"
echo "================================" | tee -a "$OUTPUT_FILE"

# Evaluate moving average smoothing with different cache parameters
# Directory structure: smooth/exp/moving_average/N{interval}O{max_order}F{first_enhance}/{alpha}
# Also supports old format: smooth/exp/moving_average/{alpha} (without cache params)
MA_ROOT="${OUTPUTS_ROOT}/smooth/exp/moving_average"

if [ -d "$MA_ROOT" ]; then
    # First, check for new format with N*O*F cache param directories
    has_cache_dirs=false
    for cache_dir in "$MA_ROOT"/N*O*F*; do
        if [ ! -d "$cache_dir" ]; then
            continue
        fi
        has_cache_dirs=true

        # Extract cache params from directory name (e.g., "N5O2F3")
        cache_params=$(basename "$cache_dir")

        # Iterate over alpha subdirectories
        for alpha_dir in "$cache_dir"/*; do
            if [ ! -d "$alpha_dir" ]; then
                continue
            fi

            alpha=$(basename "$alpha_dir")
            config_name="moving_average_${cache_params}_alpha_${alpha}"

            echo "----------------------------------------" | tee -a "$OUTPUT_FILE"
            echo "Evaluating: $config_name" | tee -a "$OUTPUT_FILE"
            echo "Cache params: $cache_params" | tee -a "$OUTPUT_FILE"
            echo "Alpha: $alpha" | tee -a "$OUTPUT_FILE"
            echo "Path: $alpha_dir" | tee -a "$OUTPUT_FILE"

            evaluate_dir "$alpha_dir" "$config_name"
        done
    done

    # If no cache param directories found, check for old format (direct alpha directories)
    if [ "$has_cache_dirs" = false ]; then
        echo "No N*O*F directories found, checking old format (direct alpha directories)..." | tee -a "$OUTPUT_FILE"
        for alpha_dir in "$MA_ROOT"/*; do
            if [ ! -d "$alpha_dir" ]; then
                continue
            fi

            alpha=$(basename "$alpha_dir")
            # Skip if it's a cache param directory
            case "$alpha" in
                N*O*F*) continue ;;
            esac

            config_name="moving_average_alpha_${alpha}"

            echo "----------------------------------------" | tee -a "$OUTPUT_FILE"
            echo "Evaluating: $config_name (old format)" | tee -a "$OUTPUT_FILE"
            echo "Alpha: $alpha" | tee -a "$OUTPUT_FILE"
            echo "Path: $alpha_dir" | tee -a "$OUTPUT_FILE"

            evaluate_dir "$alpha_dir" "$config_name"
        done
    fi
else
    echo "Moving average directory not found: $MA_ROOT" | tee -a "$OUTPUT_FILE"
fi

echo "================================"
echo "Evaluation completed! Results saved to: $OUTPUT_FILE"
echo "================================"
