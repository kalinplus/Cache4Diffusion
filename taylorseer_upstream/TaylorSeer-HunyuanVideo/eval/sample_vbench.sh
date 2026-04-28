#!/bin/bash
# 生成 VBench 的评估视频

set -euo pipefail

export MODEL_BASE=/mnt/data0/tencent/HunyuanVideo
export CUDA_VISIBLE_DEVICES='2'

# Fixed parameters
Num_Devices="1"
SEED="42"
Num_Videos_per_Sample="1"
full_info_path="./eval/"
Path2Log="/home/hkl/TaylorSeer/TaylorSeer-HunyuanVideo/results/HunyuanVideo_vbench_logs"

JSON_FILE="${full_info_path}/VBench_full_info.json"
if [ ! -f "$JSON_FILE" ]; then
    echo "Error: JSON file not found at ${JSON_FILE}"
    exit 1
fi

# Calculate the total number of prompts in the JSON file
if command -v jq &>/dev/null; then
    total_prompts=$(jq '. | length' "$JSON_FILE")
else
    echo "Warning: jq not found, using python for JSON parsing."
    total_prompts=$(python3 -c "import json; f=open('$JSON_FILE'); data=json.load(f); print(len(data))")
fi

echo "Total number of prompts: $total_prompts"

# Compute the number of prompts each device will handle (the last device may have fewer)
chunk_size=$(( (total_prompts + Num_Devices - 1) / Num_Devices ))

# Configurations: N=FRESH_THRESHOLD, O=MAX_ORDER, F=FIRST_ENHANCE, alpha=SMOOTHING_ALPHA
# TAYLOR_USE_SMOOTHING is inferred from alpha (0 → False, >0 → True)
configs=(
    # "3  1  1  0.8"
    # "5  1  1  0.8"
    "6  1  1  0.8"
)

for cfg in "${configs[@]}"; do
    read -r N O F alpha <<< "$cfg"

    Video_Save_Path="/home/hkl/TaylorSeer/TaylorSeer-HunyuanVideo/results/vbench-simulate/N${N}O${O}F${F}A${alpha}"

    # Infer smoothing from alpha
    if [ "$alpha" = "0" ]; then
        TAYLOR_USE_SMOOTHING="False"
    else
        TAYLOR_USE_SMOOTHING="True"
    fi

    echo "============================================"
    echo "Config: N=${N}, O=${O}, F=${F}, alpha=${alpha}, smoothing=${TAYLOR_USE_SMOOTHING}"
    echo "Save path: ${Video_Save_Path}"
    echo "============================================"

    export TAYLOR_FRESH_THRESHOLD="$N"
    export TAYLOR_MAX_ORDER="$O"
    export TAYLOR_FIRST_ENHANCE="$F"
    export TAYLOR_USE_SMOOTHING="$TAYLOR_USE_SMOOTHING"
    export TAYLOR_USE_HYBRID_SMOOTHING="False"
    export TAYLOR_SMOOTHING_METHOD="exponential"
    export TAYLOR_SMOOTHING_ALPHA="$alpha"

    # Ensure the log directory exists
    mkdir -p "$Path2Log"

    # Launch separate background processes for each GPU
    for (( d=0; d<Num_Devices; d++ )); do
        {
            index_start=$(( d * chunk_size ))
            index_end=$(( (d+1) * chunk_size - 1 ))
            if [ $index_end -ge $total_prompts ]; then
                index_end=$(( total_prompts - 1 ))
            fi

            log_file="${Path2Log}/device_${d}_N${N}_O${O}_F${F}_a${alpha}.log"
            echo "Device $d: Processing prompts index range [$index_start, $index_end]" > "$log_file"

            python3 sample_video_vbench.py \
                --model-base /mnt/data0/tencent/HunyuanVideo \
                --dit-weight /mnt/data0/tencent/HunyuanVideo/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
                --vbench-json-path "$JSON_FILE" \
                --index-start "$index_start" \
                --index-end "$index_end" \
                --seed "$SEED" \
                --num-videos-per-prompt "$Num_Videos_per_Sample" \
                --video-size 480 640 \
                --video-length 65 \
                --infer-steps 50 \
                --flow-reverse \
                --use-cpu-offload \
                --save-path "$Video_Save_Path" >> "$log_file" 2>&1

            echo "Device $d: Completed inference for index range [$index_start, $index_end]" >> "$log_file"
        } &
    done

    wait
    echo "Config N=${N}, O=${O}, F=${F}, alpha=${alpha} completed."
done

echo "All configurations have been completed!"
