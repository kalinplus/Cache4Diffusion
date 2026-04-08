#!/bin/bash
model_path='/mnt/data0/pretrained_models/tencent/HunyuanImage-2.1'
model_name='hunyuanimage-v2.1'
export CUDA_VISIBLE_DEVICES='1'
export HUNYUANIMAGE_V2_1_MODEL_ROOT="$model_path"

# Prompt file containing one prompt per line
prompt_file='assets/prompts/DrawBench200.txt'

lambdas=(0.6 0.8)

for lambda in "${lambdas[@]}"; do
    python HunyuanImage-2.1/run_hyimage_teacache_lite_batch.py \
        --prompt_file "$prompt_file" \
        --model_name "$model_name" \
        --seed 649151 \
        --width 2048 \
        --height 2048 \
        --shift 5 \
        --guidance_scale 3.5 \
        --outdir "outputs/hyimage2.1/teacache/without_refiner/lambda$lambda" \
        --prefix "TeaCache_lambda$lambda" \
        --use_reprompt \
        --rel_l1_thresh $lambda
        # --start_idx 0 \
        # --end_idx 5 \
        # --use_refiner \
done