#!/bin/bash
export CUDA_VISIBLE_DEVICES='0'

PROJECT_ROOT="/home/hkl/Cache4Diffusion"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}/taylorseer"

python sample.py \
    --input_image img.jpg \
    --prompt_file "prompts/DrawBench200.txt" \
    --negative_prompt "lowres, blurry, worst quality, jpeg artifacts" \
    --width 1328 \
    --height 1328 \
    --num_steps 50 \
    --guidance_scale 1.0 \
    --seed 0 \
    --num_images_per_prompt 1 \
    --batch_size 1 \
    --model_name qwen-image \
    --output_dir "samples/db200/baseline"
