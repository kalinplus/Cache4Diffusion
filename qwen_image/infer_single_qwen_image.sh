local_model_path="/mnt/data0/pretrained_models/Qwen/Qwen-Image"
model_id="Qwen/Qwen-Image"

PROJECT_ROOT="/home/hkl/Cache4Diffusion"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES='0,1'
# export TS_DEBUG_SHAPES=1
# export TS_STRICT_SHAPES=1

# cd qwen_image
python qwen_image/taylorseer_qwen_image/diffusers_taylorseer_qwen_image.py \
    --model "$local_model_path" \
    --steps 50 \
    --seed 42 \
    --dtype bfloat16 \
    --true_cfg_scale 7.5 \
    --outdir outputs \
    --prefix TaylorSeer \
    --prompt "A coffee shop entrance features a chalkboard sign reading 'Qwen Coffee 😊 $2 per cup,' with a neon light beside it displaying '通义千问'. Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written 'π≈3.1415926-53589793-23846264-33832795-02384197'." \
    --use_taylor \