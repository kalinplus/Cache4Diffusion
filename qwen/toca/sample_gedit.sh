#!/bin/bash

# GEdit Benchmark 测试脚本
# 使用本地模型路径进行 Qwen-Image-Edit 图像编辑测试 (ToCa 缓存)

export CUDA_VISIBLE_DEVICES=3

# Qwen Image Edit 模型路径（直接传本地路径给 --model_name）
MODEL_PATH="/mnt/data0/Qwen-Image-Edit-2509"

cd "$(dirname "$0")"

# ToCa 配置列表
# 格式: label interval fresh_ratio soft_fresh_weight
CONFIG_LIST=(
    "N8R70 8 0.70 0.25"
    "N12R75 12 0.75 0.25"
)

for config in "${CONFIG_LIST[@]}"; do
    read -r label interval fresh_ratio soft_fresh_weight <<< "$config"
    output_dir="/home/hkl/Cache4Diffusion/samples/GEdit/qwen-image-edit/toca/${label}"

    echo "=========================================="
    echo "ToCa Config: ${label} (interval=${interval}, fresh_ratio=${fresh_ratio})"
    echo "=========================================="

    torchrun --standalone --nproc_per_node=1 sample_gedit.py \
        --dataset_path "/mnt/data0/datasets/stepfun-ai/GEdit-Bench" \
        --model_name "$MODEL_PATH" \
        --output_dir "$output_dir" \
        --num_steps 50 \
        --guidance_scale 1.0 \
        --seed 0 \
        --interval "$interval" \
        --fresh_ratio "$fresh_ratio" \
        --soft_fresh_weight "$soft_fresh_weight" \
        --english_only
done

# 配置说明：
# N=8, R=70%: interval=8, fresh_ratio=0.70 → ~4.5-5x speedup
# N=12, R=75%: interval=12, fresh_ratio=0.75 → ~6x speedup
#
# MODEL_PATH: 直接传本地路径给 --model_name（from_pretrained 支持本地路径）
# --interval: 缓存间隔步数 N
# --fresh_ratio: token 刷新比例 R
# --soft_fresh_weight: 软刷新权重
# --english_only: 仅处理英文任务
