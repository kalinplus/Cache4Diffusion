#!/bin/bash
# Description: This script demonstrates how to inference a video based on HunyuanVideo model

export MODEL_BASE=/mnt/data0/tencent/HunyuanVideo

python3 sample_video.py \
    --model-base /mnt/data0/tencent/HunyuanVideo \
    --dit-weight /mnt/data0/tencent/HunyuanVideo/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --video-size 480 640 \
    --video-length 65 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --seed 42 \
    --flow-reverse \
    --use-cpu-offload \
    --save-path ./results
