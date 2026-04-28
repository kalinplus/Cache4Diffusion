#!/usr/bin/env bash
# FasterCache SDXL inference script

local_model_path="/mnt/data1/pretrained_models/stabilityai/stable-diffusion-xl-base-1.0"
prompt_file="/home/hkl/stable_diffusion_xl/fastercache_sdxl/prompts/DrawBench200.txt"
base_outdir="/home/hkl/stable_diffusion_xl/samples/fastercache"

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export SDXL_MODEL_PATH="$local_model_path"

# Sampling parameters
height=1024
width=1024
num_steps=50
guidance=5.0
seed=0

# Specific configurations: start_interval_alpha
configs=(
    "1 3 0.3"
    "3 8 0.3"
    "3 5 0.3"
    "15 2 0.3"
)

for config in "${configs[@]}"; do
    read -r fc_start_step fc_interval fc_alpha <<< "$config"
    echo "Running start_step=$fc_start_step, interval=$fc_interval, alpha=$fc_alpha ..."

    outdir="${base_outdir}/S${fc_start_step}_N${fc_interval}_A${fc_alpha}"

    torchrun --nproc_per_node=4 sample.py \
        --prompt_file "$prompt_file" \
        --height "$height" \
        --width "$width" \
        --num_steps "$num_steps" \
        --guidance "$guidance" \
        --seed "$seed" \
        --output_dir "$outdir" \
        --fc_start_step "$fc_start_step" \
        --fc_interval "$fc_interval" \
        --fc_alpha "$fc_alpha"

    echo "Finished start_step=$fc_start_step, interval=$fc_interval, alpha=$fc_alpha. Results saved to $outdir"
    echo "-------------S${fc_start_step}_N${fc_interval}_A${fc_alpha}-------------"
done
