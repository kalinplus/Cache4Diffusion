#!/bin/bash

# GEdit Benchmark 测试脚本
# 使用本地模型路径进行 FLUX-Kontext 图像编辑测试

export CUDA_VISIBLE_DEVICES=2

# 本地模型路径（通过环境变量注入，优先级高于 config 里的硬编码路径）
export FLUX_MODEL="/mnt/data0/pretrained_models/black-forest-labs/FLUX.1-Kontext-dev/flux1-kontext-dev.safetensors"
export FLUX_AE="/mnt/data0/pretrained_models/black-forest-labs/FLUX.1-Kontext-dev/ae.safetensors"
export T5_MODEL_PATH="/mnt/data0/pretrained_models/google/t5-v1_1-xxl"
export CLIP_MODEL_PATH="/mnt/data0/pretrained_models/openai/clip-vit-large-patch14"

# cd flux/teacache

# TeaCache 参数组合
REL_L1_THRESH=(0.8 1.0 1.2)

for rel_l1_thresh in "${REL_L1_THRESH[@]}"; do
    output_dir="/home/hkl/Cache4Diffusion/samples/GEdit/flux-kontext/teacache/R${rel_l1_thresh}"

    torchrun --standalone --nproc_per_node=1 src/sample_gedit.py \
        --dataset_path "/mnt/data0/datasets/stepfun-ai/GEdit-Bench" \
        --model_name "flux-dev-kontext" \
        --output_dir "$output_dir" \
        --num_steps 50 \
        --guidance 3.5 \
        --seed 0 \
        --rel_l1_thresh "$rel_l1_thresh" \
        --english_only
done

# 参数说明：
# model_name: 仍传 "flux-dev-kontext"（决定 FluxParams 等模型配置）
# FLUX_MODEL / FLUX_AE / T5_MODEL_PATH / CLIP_MODEL_PATH: 通过环境变量覆盖实际权重路径
# --rel_l1_thresh: TeaCache 相对 L1 阈值（越小越激进加速，默认 1.0）
# --english_only: 仅处理英文任务
