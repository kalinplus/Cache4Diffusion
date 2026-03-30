#!/bin/bash

# GEdit Benchmark 测试脚本
# 使用本地模型路径进行 FLUX-Kontext 图像编辑测试

export CUDA_VISIBLE_DEVICES=0
export FLUX_MODEL="/mnt/data0/pretrained_models/black-forest-labs/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors"
export FLUX_AE="/mnt/data0/pretrained_models/black-forest-labs/FLUX.1-Kontext-dev/ae.safetensors"
export T5_MODEL_PATH="/mnt/data0/pretrained_models/google/t5-v1_1-xxl"
export CLIP_MODEL_PATH="/mnt/data0/pretrained_models/openai/clip-vit-large-patch14"

# Smoothing parameters
USE_SMOOTHING="False"
USE_HYBRID_SMOOTHING="False"
SMOOTHING_METHOD="exponential"
PRINT_SMOOTHING_CONFIG="True"

INTERVAL=(9)
MAX_ORDERS=(1 2)
FIRST_ENHANCE=(3)
ALPHAS=(0.8)
# prompt_file="prompts/DrawBench200.txt"

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
                output_dir="/home/hkl/Cache4Diffusion/samples/GEdit/taylorseer/N${interval}O${max_order}F${first_enhance}Alpha${alpha}"

                torchrun --standalone --nproc_per_node=1 flux/taylorseer/src/sample_gedit.py \
                    --dataset_path "/mnt/data0/datasets/stepfun-ai/GEdit-Bench" \
                    --model_name "flux-dev-kontext" \
                    --output_dir "$output_dir" \
                    --num_steps 50 \
                    --guidance 3.5 \
                    --seed 0 \
                    --interval "$interval" \
                    --max_order "$max_order" \
                    --first_enhance "$first_enhance" \
                    --english_only
            done
        done
    done
done

# 参数说明：
# --dataset_path: GEdit-Bench 数据集路径
# --model_name: FLUX 模型配置名称（如 flux-dev-kontext）
# --output_dir: 输出目录
# --num_steps: 采样步数（默认 50）
# --guidance: 引导强度（默认 3.5）
# --seed: 随机种子（默认 0）
# --interval: 缓存刷新周期（默认 4，越大越快但质量可能略低）
# --max_order: Taylor 展开阶数（默认 2，越大精度越高）
# --first_enhance: 初始增强步数（默认 3，保证早期质量）
# --english_only: 仅处理英文任务（可选）
