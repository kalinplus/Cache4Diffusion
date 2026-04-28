#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES='2'
export TS_DEBUG_SMOOTH=0
export TS_DEBUG_FILTER=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export QWEN_IMAGE_MODEL_PATH="/mnt/data1/pretrained_models/Qwen/Qwen-Image"

cd "${SCRIPT_DIR}"

export SMOOTHING_METHOD="exponential"

INTERVAL=(0)
MAX_ORDERS=(0)
FIRST_ENHANCE=(50)
ALPHAS=(0)

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

                output_dir="${PROJECT_ROOT}/samples/qwen/taylorseer/db200/N${interval}O${max_order}F${first_enhance}Alpha${alpha}"

                python sample.py \
                    --input_image "${INPUT_IMAGE}" \
                    --prompt_file "${PROMPT_FILE}" \
                    --model_path "${MODEL_PATH}" \
                    --output_dir "${output_dir}" \
                    --interval "${interval}" \
                    --max_order "${max_order}" \
                    --first_enhance "${first_enhance}"
            done
        done
    done
done
