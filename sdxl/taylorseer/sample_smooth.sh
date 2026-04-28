#!/usr/bin/env bash
# TaylorSeer SDXL Smooth 批量推理脚本
# 参考 flux/taylorseer/sample_schnell.sh 风格，支持平滑参数和 alpha 循环

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="2,3"

# Optional: set local model path
export SDXL_MODEL_PATH="/mnt/data1/pretrained_models/stabilityai/stable-diffusion-xl-base-1.0"
export SDXL_VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

export TS_DEBUG_SMOOTH=0

prompt_file="prompts/DrawBench200.txt"
base_outdir="samples/taylorseer-smooth"

# Sampling parameters
height=1024
width=1024
num_steps=50
guidance=5.0
seed=0

# Cache parameters
INTERVALS=(3 5 6)
MAX_ORDERS=(1 2)
FIRST_ENHANCES=(3)

# Smoothing parameters
# alpha=0 表示禁用平滑；alpha>0 表示启用平滑并设置对应 alpha
ALPHAS=(0.75 0.8)
SMOOTHING_METHODS=(exponential)

for interval in "${INTERVALS[@]}"; do
    for max_order in "${MAX_ORDERS[@]}"; do
        for first_enhance in "${FIRST_ENHANCES[@]}"; do
            for alpha in "${ALPHAS[@]}"; do
                for method in "${SMOOTHING_METHODS[@]}"; do
                        if [ "$alpha" = "0" ]; then
                            smoothing_flag=""
                            alpha_flag=""
                            dir_label="Alpha0"
                        else
                            smoothing_flag="--use_smoothing"
                            alpha_flag="--smoothing_alpha $alpha"
                            dir_label="Alpha${alpha}"
                        fi

                        outdir="${base_outdir}/N${interval}O${max_order}F${first_enhance}_${method}_${dir_label}"

                        echo "============================================================"
                        echo "Running: interval=$interval, order=$max_order, first=$first_enhance, alpha=$alpha, method=$method"
                        echo "Output: $outdir"
                        echo "============================================================"

                        torchrun --nproc_per_node=1 sample.py \
                            --prompt_file "$prompt_file" \
                            --height "$height" \
                            --width "$width" \
                            --num_steps "$num_steps" \
                            --guidance "$guidance" \
                            --seed "$seed" \
                            --output_dir "$outdir" \
                            --interval "$interval" \
                            --max_order "$max_order" \
                            --first_enhance "$first_enhance" \
                            --smoothing_method "$method" \
                            $smoothing_flag \
                            $alpha_flag

                        echo "Finished: $outdir"
                        echo ""
                done
            done
        done
    done
done

# Optional: auto evaluate all results
echo ""
echo "===== Evaluation ====="
for interval in "${INTERVALS[@]}"; do
    for max_order in "${MAX_ORDERS[@]}"; do
        for first_enhance in "${FIRST_ENHANCES[@]}"; do
            for alpha in "${ALPHAS[@]}"; do
                if [ "$alpha" = "0" ]; then
                    dir_label="Alpha0"
                else
                    dir_label="Alpha${alpha}"
                fi
                outdir="${base_outdir}/N${interval}O${max_order}F${first_enhance}_${dir_label}"
                if [ -d "$outdir" ]; then
                    echo "--- Evaluating $outdir ---"
                    python evaluate.py --test_folder "$outdir"
                fi
            done
        done
    done
done
