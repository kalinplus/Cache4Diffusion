#!/bin/bash
export CUDA_VISIBLE_DEVICES='5'

PROJECT_ROOT="/home/hkl/Cache4Diffusion"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}/taylorseer"

INTERVAL=(6)  # 4 5 6
MAX_ORDERS=(2)  # 1 2
FIRST_ENHANCE=(3)
ALPHAS=(0.7 0.75 0.8 0.85 0.9)
prompt_file="prompts/DrawBench200.txt"

for interval in "${INTERVAL[@]}"; do
    for max_order in "${MAX_ORDERS[@]}"; do
        for first_enhance in "${FIRST_ENHANCE[@]}"; do
            for alpha in "${ALPHAS[@]}"; do
                output_dir="samples/db200/smooth/N${interval}O${max_order}F${first_enhance}Alpha${alpha}"
                python sample.py \
                    --input_image img.jpg \
                    --prompt_file "${prompt_file}" \
                    --negative_prompt "lowres, blurry, worst quality, jpeg artifacts" \
                    --width 1328 \
                    --height 1328 \
                    --num_steps 50 \
                    --guidance_scale 1.0 \
                    --seed 0 \
                    --num_images_per_prompt 1 \
                    --batch_size 1 \
                    --model_name qwen-image \
                    --output_dir "${output_dir}" \
                    --monitor_gpu_usage \
                    --use_taylor \
                    --interval "${interval}" \
                    --max_order "${max_order}" \
                    --first_enhance "${first_enhance}" \
                    --use_smoothing \
                    --smoothing_method "exponential" \
                    --smoothing_alpha "${alpha}"
            done
        done
    done
done