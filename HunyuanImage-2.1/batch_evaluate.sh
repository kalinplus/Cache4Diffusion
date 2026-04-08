#!/bin/bash
export CUDA_VISIBLE_DEVICES='3'

PROJECT_ROOT="/home/hkl/Cache4Diffusion"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

PROMPT_FILE="assets/prompts/DrawBench200.txt"
OUTPUTS_ROOT="/home/hkl/Cache4Diffusion/outputs"

# Reference folder: 50-step baseline
REFERENCE_FOLDER="${OUTPUTS_ROOT}/HunyuanImage2.1/origin/without_refiner/50steps"
CLIP_MODEL_PATH="/mnt/data0/pretrained_models/laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
IMAGEREWARD_MODEL_PATH="/mnt/data0/pretrained_models/zai-org/ImageReward"

# Function to evaluate a single directory
evaluate_dir() {
    local test_dir=$1
    local config_name=$2
    local out_file=$3

    if [ ! -d "$test_dir" ]; then
        echo "[SKIP] $config_name (directory not found: $test_dir)" | tee -a "$out_file"
        return
    fi

    if [ -z "$(ls -A "$test_dir" 2>/dev/null)" ]; then
        echo "[SKIP] $config_name (empty directory)" | tee -a "$out_file"
        return
    fi

    echo "" | tee -a "$out_file"
    echo "----------------------------------------" | tee -a "$out_file"
    echo "Config: $config_name" | tee -a "$out_file"
    echo "Path:  $test_dir" | tee -a "$out_file"
    echo "----------------------------------------" | tee -a "$out_file"

    python HunyuanImage-2.1/evaluate.py \
        --test_folder "$test_dir" \
        --prompt_file "$PROMPT_FILE" \
        --reference_folder "$REFERENCE_FOLDER" \
        --clip_model_path "$CLIP_MODEL_PATH" \
        --imagereward_model_path "$IMAGEREWARD_MODEL_PATH" \
        | tee -a "$out_file"

    echo "" | tee -a "$out_file"
}

# ============================================================
# Group 0: Step Reduction baselines
# Reference: origin/without_refiner/50steps
# Output:  origin/without_refiner/evaluation_results.txt
# ============================================================
ORIGIN_RESULT="${OUTPUTS_ROOT}/HunyuanImage2.1/origin/without_refiner/evaluation_results.txt"
> "$ORIGIN_RESULT"

echo "========================================" | tee -a "$ORIGIN_RESULT"
echo "Group 0: Step Reduction" | tee -a "$ORIGIN_RESULT"
echo "Reference: $REFERENCE_FOLDER" | tee -a "$ORIGIN_RESULT"
echo "========================================" | tee -a "$ORIGIN_RESULT"

evaluate_dir "${OUTPUTS_ROOT}/HunyuanImage2.1/origin/without_refiner/10steps" "origin_10steps" "$ORIGIN_RESULT"
evaluate_dir "${OUTPUTS_ROOT}/HunyuanImage2.1/origin/without_refiner/17steps" "origin_17steps" "$ORIGIN_RESULT"
evaluate_dir "${OUTPUTS_ROOT}/HunyuanImage2.1/origin/without_refiner/34steps" "origin_34steps" "$ORIGIN_RESULT"

echo "Done! Results saved to: $ORIGIN_RESULT" | tee -a "$ORIGIN_RESULT"

# ============================================================
# Group 1: FORA (TaylorSeer max_order=0)
# Reference: origin/without_refiner/50steps
# Output:  HunyuanImage2.1/FORA/evaluation_results.txt
# ============================================================
FORA_ROOT="${OUTPUTS_ROOT}/HunyuanImage2.1/FORA"
FORA_RESULT="${FORA_ROOT}/evaluation_results.txt"
> "$FORA_RESULT"

echo "========================================" | tee -a "$FORA_RESULT"
echo "Group 1: FORA" | tee -a "$FORA_RESULT"
echo "Reference: $REFERENCE_FOLDER" | tee -a "$FORA_RESULT"
echo "========================================" | tee -a "$FORA_RESULT"

if [ -d "$FORA_ROOT" ]; then
    for dir in "$FORA_ROOT"/*; do
        if [ ! -d "$dir" ]; then
            continue
        fi
        config_name="FORA_$(basename "$dir")"
        evaluate_dir "$dir" "$config_name" "$FORA_RESULT"
    done
else
    echo "[SKIP] FORA directory not found: $FORA_ROOT" | tee -a "$FORA_RESULT"
fi

echo "Done! Results saved to: $FORA_RESULT" | tee -a "$FORA_RESULT"

# ============================================================
# Group 2: TeaCache
# Reference: origin/without_refiner/50steps
# Output:  HunyuanImage2.1/TeaCache/without_refiner/evaluation_results.txt
# ============================================================
TEACACHE_ROOT="${OUTPUTS_ROOT}/HunyuanImage2.1/TeaCache/without_refiner"
TEACACHE_RESULT="${TEACACHE_ROOT}/evaluation_results.txt"
> "$TEACACHE_RESULT"

echo "========================================" | tee -a "$TEACACHE_RESULT"
echo "Group 2: TeaCache" | tee -a "$TEACACHE_RESULT"
echo "Reference: $REFERENCE_FOLDER" | tee -a "$TEACACHE_RESULT"
echo "========================================" | tee -a "$TEACACHE_RESULT"

if [ -d "$TEACACHE_ROOT" ]; then
    for dir in "$TEACACHE_ROOT"/*; do
        if [ ! -d "$dir" ]; then
            continue
        fi
        config_name="TeaCache_$(basename "$dir")"
        evaluate_dir "$dir" "$config_name" "$TEACACHE_RESULT"
    done
else
    echo "[SKIP] TeaCache directory not found: $TEACACHE_ROOT" | tee -a "$TEACACHE_RESULT"
fi

echo "Done! Results saved to: $TEACACHE_RESULT" | tee -a "$TEACACHE_RESULT"
