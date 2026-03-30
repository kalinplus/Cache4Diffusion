#!/bin/bash

# GEdit Benchmark 测试脚本
# 使用本地模型路径进行 Qwen-Image-Edit 图像编辑测试

export CUDA_VISIBLE_DEVICES=2

# Qwen Image Edit 模型路径（直接传本地路径给 --model_name）
MODEL_PATH="/mnt/data0/Qwen-Image-Edit-2509"

# TeaCache 参数组合
REL_L1_THRESH=(1.0 1.2 1.4)

cd "$(dirname "$0")"

for rel_l1_thresh in "${REL_L1_THRESH[@]}"; do
    output_dir="/home/hkl/Cache4Diffusion/samples/GEdit/qwen-image-edit/teacache/R${rel_l1_thresh}"

    torchrun --standalone --nproc_per_node=1 sample_gedit.py \
        --dataset_path "/mnt/data0/datasets/stepfun-ai/GEdit-Bench" \
        --model_name "$MODEL_PATH" \
        --output_dir "$output_dir" \
        --num_steps 50 \
        --guidance_scale 1.0 \
        --seed 0 \
        --rel_l1_thresh "$rel_l1_thresh" \
        --enable_teacache \
        --english_only
done

# 参数说明：
# MODEL_PATH: 直接传本地路径给 --model_name（from_pretrained 支持本地路径）
# --enable_teacache: 启用 TeaCache 加速
# --rel_l1_thresh: TeaCache 相对 L1 阈值（越小越激进加速，默认 1.0）
# --english_only: 仅处理英文任务
