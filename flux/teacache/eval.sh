#!/bin/bash
# NOTE: 未适配具体代码
# Evaluate teacache generated images using CLIP Score, ImageReward, PSNR, SSIM, LPIPS

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

# Paths
TEST_FOLDER="outputs/teacache"
REFERENCE_FOLDER="outputs/origin"
PROMPT_FILE="prompts/DrawBench200.txt"

# Model paths
CLIP_MODEL_PATH="/data/public/.cache/huggingface/hub/models--laion--CLIP-ViT-g-14-laion2B-s12B-b42K/snapshots/4b0305adc6802b2632e11cbe6606a9bdd43d35c9"
IMAGEREWARD_MODEL_PATH="/data/public/.cache/huggingface/hub/models--zai-org--ImageReward/snapshots/5736be03b2652728fb87788c9797b0570450ab72"

echo "=========================================="
echo "Evaluating teacache outputs"
echo "Test folder: $TEST_FOLDER"
echo "Reference folder: $REFERENCE_FOLDER"
echo "=========================================="

python evaluate.py \
    --test_folder "$TEST_FOLDER" \
    --prompt_file "$PROMPT_FILE" \
    --reference_folder "$REFERENCE_FOLDER" \
    --clip_model_path "$CLIP_MODEL_PATH" \
    --imagereward_model_path "$IMAGEREWARD_MODEL_PATH"

# 参数说明：
# --test_folder: 测试图片目录（由 sample 脚本生成）
# --prompt_file: 文本提示文件（默认 prompts/DrawBench200.txt）
# --reference_folder: 参考图片目录（用于计算 PSNR/SSIM/LPIPS）
# --clip_model_path: CLIP 模型路径（用于计算 CLIP Score）
# --imagereward_model_path: ImageReward 模型路径（用于计算 ImageReward）
#
# 输出格式：
#   Result:(ClipScore, ImageReward, PSNR, SSIM, LPIPS)
