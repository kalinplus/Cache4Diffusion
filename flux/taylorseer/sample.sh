#!/bin/bash

# FLUX-Kontext 推理脚本（基于 sample_gedit.sh 的参数风格）
# 用于运行 src/sample.py（单机单卡）

set -euo pipefail

export CUDA_VISIBLE_DEVICES=2

# 请按需修改以下路径
PROMPT_FILE="prompt.txt"   # 每行一个 prompt
INPUT_IMAGE="img.jpg"     # Kontext 条件图
MODEL_PATH="/mnt/data0/pretrained_models/black-forest-labs/FLUX.1-Kontext-dev"  # 本地模型目录或 .safetensors
OUTPUT_DIR="samples/test"

python src/sample.py \
    --prompt_file "${PROMPT_FILE}" \
    --input_image "${INPUT_IMAGE}" \
    --model_name "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --width 1360 \
    --height 768 \
    --num_steps 50 \
    --guidance 3.5 \
    --seed 0 \
    --num_images_per_prompt 1 \
    --batch_size 1 \
    --interval 4 \
    --max_order 2 \
    --first_enhance 3 \
    --add_sampling_metadata

# 参数说明：
# --prompt_file: prompt 文本路径（每行一个）
# --input_image: 输入参考图路径（Kontext 模式必需）
# --model_name: 支持配置名或本地模型路径
# --num_steps / --guidance / --seed: 采样与随机参数
# --interval / --max_order / --first_enhance: TaylorSeer 缓存相关参数
#
# 注意：
# src/sample.py 不支持 --t5_path / --clip_path 参数，
# T5/CLIP 路径由 src/flux/util.py 中 load_t5/load_clip 决定。
