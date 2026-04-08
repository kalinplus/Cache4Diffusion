#!/bin/bash
model_path='/mnt/data0/pretrained_models/tencent/HunyuanImage-2.1'
model_name='hunyuanimage-v2.1'
export CUDA_VISIBLE_DEVICES='2'
export HUNYUANIMAGE_V2_1_MODEL_ROOT="$model_path"

# Prompt file containing one prompt per line
prompt_file='assets/prompts/DrawBench200.txt'

python HunyuanImage-2.1/run_hyimage_taylorseer_lite_batch.py \
    --prompt_file "$prompt_file" \
    --model_name "$model_name" \
    --seed 649151 \
    --width 2048 \
    --height 2048 \
    --shift 5 \
    --guidance_scale 3.5 \
    --outdir outputs/naive_ts/without_refiner \
    --prefix TaylorSeer \
    --use_reprompt \
    --use_taylorseer_lite
    # --start_idx 0 \
    # --end_idx 5 \
    # --use_refiner \
