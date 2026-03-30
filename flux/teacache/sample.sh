#!/bin/bash

# FLUX-Kontext 推理脚本（单机单卡）

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

# 本地模型路径（通过环境变量注入，优先级高于 config 里的硬编码路径）
export FLUX_MODEL="/mnt/data0/pretrained_models/black-forest-labs/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors"
export FLUX_AE="/mnt/data0/pretrained_models/black-forest-labs/FLUX.1-Kontext-dev/ae.safetensors"
export T5_MODEL_PATH="/mnt/data0/pretrained_models/google/t5-v1_1-xxl"
export CLIP_MODEL_PATH="/mnt/data0/pretrained_models/openai/clip-vit-large-patch14"

# 采样参数
PROMPT_FILE="prompt.txt"       # 每行一个 prompt
INPUT_IMAGE="img.jpg"          # Kontext 条件图
MASK_PATH="mask.jpg"           # mask 图片（可选，用于 fill 模式）
MODEL_NAME="flux-dev-kontext"   # 模型配置名（决定 FluxParams 等配置）
OUTPUT_DIR="outputs/teacache"

# TeaCache 参数
REL_L1_THRESH="1.0"

python src/sample.py \
    --prompt_file "${PROMPT_FILE}" \
    --input_image "${INPUT_IMAGE}" \
    --mask_path "${MASK_PATH}" \
    --model_name "${MODEL_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --width 1360 \
    --height 768 \
    --num_steps 50 \
    --guidance 3.5 \
    --seed 0 \
    --num_images_per_prompt 1 \
    --batch_size 1 \
    --enable_teacache \
    --rel_l1_thresh "${REL_L1_THRESH}" \
    --add_sampling_metadata

# 参数说明：
# model_name: 仍传 "flux-dev-kontext"（决定 FluxParams 等模型配置）
# FLUX_MODEL / FLUX_AE / T5_MODEL_PATH / CLIP_MODEL_PATH: 通过环境变量覆盖实际权重路径
# --enable_teacache: 启用 TeaCache
# --rel_l1_thresh: TeaCache L1 阈值（越小越激进加速，默认 1.0）
