#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES='0,1,2,3'
export TS_DEBUG_SMOOTH=1
export TS_DEBUG_FILTER="0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="/home/hkl/calculate-flops.pytorch:${PROJECT_ROOT}/qwen/taylorseer:${PYTHONPATH:-}"
export QWEN_IMAGE_MODEL_PATH="/mnt/data1/pretrained_models/Qwen/Qwen-Image"

cd "${SCRIPT_DIR}"

export USE_SMOOTHING="True"
export SMOOTHING_METHOD="exponential"

INTERVAL=(6)
MAX_ORDERS=(2)
FIRST_ENHANCE=(3)
ALPHAS=(0.8)

PROMPT_FILE="${SCRIPT_DIR}/prompts/DrawBench200.txt"
INPUT_IMAGE="${PROJECT_ROOT}/qwen/teacache/img.jpg"
MODEL_PATH="${QWEN_IMAGE_MODEL_PATH}"

for interval in "${INTERVAL[@]}"; do
    for max_order in "${MAX_ORDERS[@]}"; do
        for first_enhance in "${FIRST_ENHANCE[@]}"; do
            for alpha in "${ALPHAS[@]}"; do
                if [ "$alpha" = 0 ]; then
                    export USE_SMOOTHING="False"
                    unset SMOOTHING_ALPHA
                else
                    export USE_SMOOTHING="True"
                    export SMOOTHING_ALPHA="$alpha"
                fi

                output_dir="${PROJECT_ROOT}/samples/qwen/taylorseer/benchmark_db200/N${interval}O${max_order}F${first_enhance}Alpha${alpha}"

                python benchmark_sample.py \
                    --input_image "${INPUT_IMAGE}" \
                    --prompt_file "${PROMPT_FILE}" \
                    --model_path "${MODEL_PATH}" \
                    --output_dir "${output_dir}" \
                    --num_warmup_prompts 1 \
                    --num_benchmark_prompts 10 \
                    --num_flops_prompts 1 \
                    --benchmark_report benchmark.txt \
                    --test_FLOPs \
                    --monitor_gpu_usage \
                    --interval "${interval}" \
                    --max_order "${max_order}" \
                    --first_enhance "${first_enhance}"
            done
        done
    done
done
