#!/usr/bin/env bash
# Usage: MODEL_NAME=flux bash infer.sh
#        MODEL_NAME=qwen_image bash infer.sh
#        MODEL_NAME=hunyuan_video bash infer.sh

MODEL_NAME="${MODEL_NAME:-hunyuan_video}"

if [ "$MODEL_NAME" = "flux" ]; then
    model_path="/mnt/data0/pretrained_models/black-forest-labs/FLUX.1-dev"
    dtype="float16"
    extra_args=""
elif [ "$MODEL_NAME" = "qwen_image" ]; then
    model_path="/mnt/data0/pretrained_models/Qwen/Qwen-Image"
    dtype="bfloat16"
    extra_args=""
elif [ "$MODEL_NAME" = "hunyuan_video" ]; then
    model_path="/mnt/data0/pretrained_models/hunyuanvideo-community/HunyuanVideo"  # TODO: fill in model path
    dtype="bfloat16"
    extra_args="--video_length 19 --video_size 544 960 --fps 4"
else
    echo "Unknown MODEL_NAME: $MODEL_NAME (supported: flux, qwen_image, hunyuan_video)"
    exit 1
fi

export CUDA_VISIBLE_DEVICES='5,7'
export PYTHONPATH="/home/hkl/Cache4Diffusion:${PYTHONPATH:-}"
export DIFFUSERS_ATTN_BACKEND=flash

/home/hkl/miniconda3/envs/qwenimage/bin/python infer.py \
    --model "$model_path" \
    --model_name "$MODEL_NAME" \
    --steps 50 \
    --seed 42 \
    --dtype "$dtype" \
    --guidance_scale 7.5 \
    --outdir outputs \
    --prefix "TaylorSeer_$MODEL_NAME" \
    --prompt "A beautiful painting of a sunset over a calm ocean, with a small boat in the foreground, and a few clouds in the sky. The colors are warm and the lighting is soft, creating a serene and peaceful atmosphere. The painting is detailed and the brush strokes are visible, adding to the realism of the scene." \
    $extra_args
