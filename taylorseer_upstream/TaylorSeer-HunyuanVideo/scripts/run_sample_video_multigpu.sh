#!/bin/bash
# Description: This script demonstrates how to inference a video based on HunyuanVideo model (multi-GPU)

export TOKENIZERS_PARALLELISM=false
export MODEL_BASE=/mnt/data0/tencent/HunyuanVideo

export NPROC_PER_NODE=8
export ULYSSES_DEGREE=8
export RING_DEGREE=1

torchrun --nproc_per_node=$NPROC_PER_NODE sample_video.py \
    --model-base /mnt/data0/tencent/HunyuanVideo \
    --dit-weight /mnt/data0/tencent/HunyuanVideo/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --video-size 480 640 \
    --video-length 65 \
    --infer-steps 50 \
    --seed 42 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --ulysses-degree=$ULYSSES_DEGREE \
    --ring-degree=$RING_DEGREE \
    --save-path ./results
