local_model_path="/data/public/models/FLUX.1-dev"
model_id="black-forest-labs/FLUX.1-dev"
outdir="/data/huangkailin-20250908/Cache4Diffusion/flux/outputs/smooth/db200/naive_ts"
prompt_file="/data/huangkailin-20250908/Cache4Diffusion/assets/prompts/DrawBench200.txt"
cd flux
export CUDA_VISIBLE_DEVICES='5'
python taylorseer_flux/batch_infer.py \
    --model "$local_model_path" \
    --steps 50 \
    --seed 42 \
    --dtype float16 \
    --guidance_scale 7.5 \
    --outdir "$outdir" \
    --prefix ts_smooth \
    --prompt_file "$prompt_file" \