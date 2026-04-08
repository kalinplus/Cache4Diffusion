#!/bin/bash

export CUDA_VISIBLE_DEVICES='0'
export DIFFUSERS_ATTN_BACKEND="flash"
export TOKENIZERS_PARALLELISM=false
echo $CUDA_VISIBLE_DEVICES

model="/mnt/data0/pretrained_models/hunyuanvideo-community/HunyuanVideo"
prompt_file="/home/hkl/Cache4Diffusion/assets/prompts/test.txt"

python hunyuan_video/taylorseer_hunyuan_video/batch_infer.py \
    --prompt_file "$prompt_file" \
    --video-length 129 \
    --video-size 720 1280 \
    --infer-steps 50 \
    --model "$model" \
    --dtype bfloat16 \
    --use_taylor \