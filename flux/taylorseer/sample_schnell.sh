#!/bin/bash

# FLUX-schnell TaylorSeer 批量推理脚本
# 模仿 sample_gedit.sh 风格，适配 schnell 特有约束（4步、无 guidance）

export CUDA_VISIBLE_DEVICES=5
export FLUX_MODEL="/mnt/data0/pretrained_models/black-forest-labs/FLUX.1-schnell/flux1-schnell.safetensors"
export FLUX_AE="/mnt/data0/pretrained_models/black-forest-labs/FLUX.1-schnell/ae.safetensors"
export T5_MODEL_PATH="/mnt/data0/pretrained_models/google/t5-v1_1-xxl"
export CLIP_MODEL_PATH="/mnt/data0/pretrained_models/openai/clip-vit-large-patch14"

# Smoothing parameters
USE_SMOOTHING="False"
USE_HYBRID_SMOOTHING="False"
SMOOTHING_METHOD="exponential"
PRINT_SMOOTHING_CONFIG="True"

# INTERVAL=(2 4)
# MAX_ORDERS=(0 1 2)
# FIRST_ENHANCE=(1)
# ALPHAS=(0.8 0)
INTERVAL=(2)
MAX_ORDERS=(0)
FIRST_ENHANCE=(50)
ALPHAS=(0)
PROMPT_FILE="assets/prompts/DrawBench200.txt"

for interval in "${INTERVAL[@]}"; do
    for max_order in "${MAX_ORDERS[@]}"; do
        for first_enhance in "${FIRST_ENHANCE[@]}"; do
            for alpha in "${ALPHAS[@]}"; do
                if [ "$alpha" = 0 ]; then
                    export USE_SMOOTHING="False"
                else
                    export USE_SMOOTHING="True"
                    export SMOOTHING_ALPHA="$alpha"
                fi
                output_dir="/home/hkl/Cache4Diffusion/samples/flux-schnell/taylorseer/N${interval}O${max_order}F${first_enhance}Alpha${alpha}"

                torchrun --standalone --nproc_per_node=1 flux/taylorseer/src/sample.py \
                    --prompt_file "${PROMPT_FILE}" \
                    --model_name "flux-schnell" \
                    --output_dir "$output_dir" \
                    --num_steps 4 \
                    --seed 0 \
                    --interval "$interval" \
                    --max_order "$max_order" \
                    --first_enhance "$first_enhance"
            done
        done
    done
done

# 参数说明：
# --model_name: "flux-schnell"（固定，会强制 num_steps=4、T5 maxlen=256、无 guidance）
# --prompt_file: prompt 文件路径（每行一条）
# --num_steps: 必须为 4（代码 assert 强制）
# --interval: 缓存刷新周期（默认 4；schnell 只有 4 步，设 2 可多一次 full step）
# --max_order: Taylor 展开阶数（0=FORA 零阶近似，1=一阶）
# --first_enhance: 初始增强步数（默认 1；建议 ≤2，否则缓存无意义）
# --guidance: 未传（schnell 模型完全忽略 guidance）
