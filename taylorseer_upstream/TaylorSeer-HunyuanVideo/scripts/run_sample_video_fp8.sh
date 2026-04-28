#!/bin/bash
# Description: This script demonstrates how to inference a video based on HunyuanVideo model with FP8

export MODEL_BASE=/mnt/data0/tencent/HunyuanVideo

python3 sample_video.py \
    --model-base /mnt/data0/tencent/HunyuanVideo \
    --dit-weight /mnt/data0/tencent/HunyuanVideo/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt \
    --video-size 480 640 \
    --video-length 65 \
    --infer-steps 50 \
    --seed 42 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse \
    --use-cpu-offload \
    --use-fp8 \
    --save-path ./results
