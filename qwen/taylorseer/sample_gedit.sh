#!/bin/bash

# GEdit Benchmark 测试脚本
# TaylorSeer (ts) 和 TaylorSeer-Smooth (ts-smooth) 配置
# 用于复用 freqca 数据的 N, O, Alpha 组合

export CUDA_VISIBLE_DEVICES=7

# Qwen Image Edit 模型路径（直接传本地路径给 --model_name）
MODEL_PATH="/mnt/data0/Qwen-Image-Edit-2509"

cd "$(dirname "$0")"

# TaylorSeer 参数组合
# N (interval): 5, 6, 9
# O (max_order): 1, 2
# Alpha: 0 (no smooth), 0.8 (smooth)
INTERVAL=(5 6 9)
MAX_ORDERS=(2)
ALPHAS=(0.8)
FIRST_ENHANCE=3

for interval in "${INTERVAL[@]}"; do
    for max_order in "${MAX_ORDERS[@]}"; do
        for alpha in "${ALPHAS[@]}"; do
            if [ "$alpha" = 0 ]; then
                export USE_SMOOTHING="False"
            else
                export USE_SMOOTHING="True"
                export SMOOTHING_ALPHA="$alpha"
            fi
            export USE_HYBRID_SMOOTHING="False"
            export SMOOTHING_METHOD="exponential"

            output_dir="/home/hkl/Cache4Diffusion/samples/GEdit/qwen-image-edit/taylorseer/N${interval}O${max_order}F${FIRST_ENHANCE}A${alpha}"

            echo "=========================================="
            echo "TaylorSeer: interval=${interval}, max_order=${max_order}, first_enhance=${FIRST_ENHANCE}, alpha=${alpha}"
            echo "Output: ${output_dir}"
            echo "=========================================="

            torchrun --standalone --nproc_per_node=1 sample_gedit.py \
                --dataset_path "/mnt/data0/datasets/stepfun-ai/GEdit-Bench" \
                --model_name "$MODEL_PATH" \
                --output_dir "$output_dir" \
                --num_steps 50 \
                --guidance_scale 1.0 \
                --seed 0 \
                --interval "$interval" \
                --max_order "$max_order" \
                --first_enhance "$FIRST_ENHANCE" \
                --english_only
        done
    done
done

# 参数说明：
# N=5,6,9: 缓存刷新周期（越小越慢但质量越高）
# O=1,2: Taylor 展开阶数（越大精度越高）
# Alpha=0: 禁用平滑（ts baseline）
# Alpha=0.8: 启用平滑（ts-smooth）
# first_enhance=3: 初始 3 步强制 full compute
