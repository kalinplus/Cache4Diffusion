#!/usr/bin/env bash
# TaylorSeer SDXL inference script

local_model_path="/mnt/data1/pretrained_models/stabilityai/stable-diffusion-xl-base-1.0"
prompt_file="/home/hkl/stable_diffusion_xl/taylorseer/prompts/DrawBench200.txt"
base_outdir="/home/hkl/stable_diffusion_xl/samples/taylorseer"

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export SDXL_MODEL_PATH="$local_model_path"

# Sampling parameters
height=1024
width=1024
guidance=5.0
seed=0

# Cache parameters
num_steps_list=(50 16 10)
intervals=(0)
max_order=(0)
first_enhance=50

for interval in "${intervals[@]}"; do
    for order in "${max_order[@]}"; do
        echo "Running interval=$interval, order=$order ..."

        for num_steps in "${num_steps_list[@]}"; do
            outdir="${base_outdir}/S${num_steps}/N${interval}O${order}F${first_enhance}"

            torchrun --nproc_per_node=4 sample.py \
                --prompt_file "$prompt_file" \
                --height "$height" \
                --width "$width" \
                --num_steps "$num_steps" \
                --guidance "$guidance" \
                --seed "$seed" \
                --output_dir "$outdir" \
                --interval "$interval" \
                --max_order "$order" \
                --first_enhance "$first_enhance"

            echo "Finished interval=$interval, order=$order, num_steps=$num_steps. Results saved to $outdir"
            echo "-------------S${num_steps}N${interval}O${order}F${first_enhance}-------------"
        done
    done
done

# Optional: auto evaluate all results
# echo ""
# echo "===== Evaluation ====="
# for interval in "${intervals[@]}"; do
#     outdir="${base_outdir}/N${interval}"
#     if [ -d "$outdir" ]; then
#         echo "--- N${interval} ---"
#         python evaluate.py --test_folder "$outdir"
#     fi
# done
