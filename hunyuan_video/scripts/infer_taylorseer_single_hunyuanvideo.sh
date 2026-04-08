#!/bin/bash

export CUDA_VISIBLE_DEVICES='1'
export TOKENIZERS_PARALLELISM=false
export DIFFUSERS_ATTN_BACKEND="flash"

echo $CUDA_VISIBLE_DEVICES

model="/mnt/data0/pretrained_models/hunyuanvideo-community/HunyuanVideo"

python hunyuan_video/taylorseer_hunyuan_video/diffusers_taylorseer_hunyuan_video.py \
    --prompt "A cat walks on the grass, realistic style." \
    --video-length 129 \
    --video-size 544 960 \
    --infer-steps 50 \
    --model "$model" \
    --dtype bfloat16 \
    --use_taylor \