#!/bin/bash
# Bash script for running FLUX TaylorSeer inference with cached models
# Parameters are hardcoded for easy reuse - modify values directly as needed
export CUDA_VISIBLE_DEVICES="3"

# ============================================================================
# Configuration - Model Paths (cached folder)
# ============================================================================
CACHE_DIR="/data/public/models/FLUX.1-dev"

export FLUX_MODEL="${CACHE_DIR}/flux1-dev.safetensors"
export FLUX_AE="${CACHE_DIR}/ae.safetensors"

T5_MODEL_DIR="${CACHE_DIR}/models--google--t5-v1_1-xxl"
CLIP_MODEL_DIR="/data/public/models/openai/clip-vit-large-patch14"

mkdir -p "${CACHE_DIR}/.cache/huggingface/hub"
ln -sfn "${T5_MODEL_DIR}" "${CACHE_DIR}/.cache/huggingface/hub/models--google--t5-v1_1-xxl" 2>/dev/null || true

# ============================================================================
# Inference Parameters - Modify these values as needed
# ============================================================================
MODEL_NAME="flux-dev"
WIDTH="1024"
HEIGHT="1024"
NUM_STEPS="50"
GUIDANCE="3.5"
SEED="42"
NUM_IMAGES="1"
BATCH_SIZE="1"
OUTPUT_DIR="outputs"
PROMPT_FILE="/data/huangkailin-20250908/Cache4Diffusion/assets/prompts/test.txt"

# TaylorSeer parameters
INTERVAL="4"
MAX_ORDER="0"
FIRST_ENHANCE="1"

# Smoothing parameters
USE_SMOOTHING="False"
USE_HYBRID_SMOOTHING="False"
SMOOTHING_METHOD="exponential"
SMOOTHING_ALPHA="0.8"
PRINT_SMOOTHING_CONFIG="True"

# GPU Configuration
GPU_ID="0"

# Optional flags
TEST_FLOPS=""
MONITOR_GPU=""
ADD_METADATA=""
NSFW_FILTER=""

# ============================================================================
# Run the Script
# ============================================================================
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export USE_SMOOTHING
export USE_HYBRID_SMOOTHING
export SMOOTHING_METHOD
export SMOOTHING_ALPHA
export PRINT_SMOOTHING_CONFIG
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/src"

echo "=============================================="
echo "FLUX TaylorSeer Inference"
echo "=============================================="
echo "Model:         ${MODEL_NAME}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "Image size:    ${WIDTH}x${HEIGHT}"
echo "Steps:         ${NUM_STEPS}"
echo "Guidance:      ${GUIDANCE}"
echo "Seed:          ${SEED}"
echo "GPU:           ${GPU_ID}"
echo "Flux model:    ${FLUX_MODEL}"
echo "AE model:      ${FLUX_AE}"
echo "T5 model:      ${T5_MODEL_DIR}"
echo "=============================================="

export T5_MODEL_PATH="${T5_MODEL_DIR}"
export CLIP_MODEL_PATH="${CLIP_MODEL_DIR}"

cd "${SCRIPT_DIR}/src"

python sample.py \
    --prompt_file "${PROMPT_FILE}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --num_steps "${NUM_STEPS}" \
    --guidance "${GUIDANCE}" \
    --seed "${SEED}" \
    --num_images_per_prompt "${NUM_IMAGES}" \
    --batch_size "${BATCH_SIZE}" \
    --model_name "${MODEL_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --interval "${INTERVAL}" \
    --max_order "${MAX_ORDER}" \
    --first_enhance "${FIRST_ENHANCE}" \
    ${TEST_FLOPS:-} \
    ${MONITOR_GPU:-} \
    ${ADD_METADATA:-} \
    ${NSFW_FILTER:-}

echo "Done! Images saved to ${OUTPUT_DIR}"
