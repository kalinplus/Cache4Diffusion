#!/bin/bash
model_path='/mnt/data0/pretrained_models/tencent/HunyuanImage-2.1'
model_name='hunyuanimage-v2.1'
export CUDA_VISIBLE_DEVICES='1'
export HUNYUANIMAGE_V2_1_MODEL_ROOT="$model_path"

python HunyuanImage-2.1/run_hyimage_taylorseer_lite.py \
    --model_name "$model_name" \
    --seed 649151 \
    --width 2048 \
    --height 2048 \
    --shift 5 \
    --guidance_scale 3.5 \
    --outdir outputs \
    --prefix TaylorSeer \
    --prompt "A cute, cartoon-style anthropomorphic penguin plush toy with fluffy fur, standing in a painting studio, wearing a red knitted scarf and a red beret with the word \"Tencent\" on it, holding a paintbrush with a focused expression as it paints an oil painting of the Mona Lisa, rendered in a photorealistic photographic style." \
    --use_taylorseer_lite
    # --use_reprompt \
    # --use_refiner \