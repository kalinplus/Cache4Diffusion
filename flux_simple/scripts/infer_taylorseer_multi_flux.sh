#!/usr/bin/env bash
local_model_path="/mnt/data0/pretrained_models/black-forest-labs/FLUX.1-dev"
model_id="black-forest-labs/FLUX.1-dev"

base_outdir="/home/hkl/Cache4Diffusion/flux_simple/outputs"
prompt_file="/home/hkl/Cache4Diffusion/assets/prompts/DrawBench200.txt"

# 支持的 dtype: float16, bfloat16, float8, float32
dtype="float16"

# 在线量化: none / nf4
quantize="none"

# NF4 单文件 transformer: 设置路径启用，留空则使用默认 diffusers 格式加载
# 注意: --transformer_file 与 --quantize 不能同时启用
# transformer_file="/path/to/flux1-dev-bnb-nf4-v2.safetensors"
transformer_file=""

# LoRA: 设置路径启用，留空禁用
# 黑白漫画风格 LoRA
# lora_path="/mnt/data0/pretrained_models/flux_lora/glif-anime-blockprint-style/bwmanga.safetensors"
# 动漫风格 LoRA
lora_path="/mnt/data0/pretrained_models/flux_lora/nerijs-animation2k-flux/animation2k_v1.safetensors"
# 写实风格 LoRA
# lora_path="/mnt/data0/pretrained_models/flux_lora/realism_lora.safetensors"
# flux-anime LoRA
# lora_path="/mnt/data0/pretrained_models/flux_lora/anime_lora.safetensors"
lora_scale=1.0

cd flux_simple
export PYTHONPATH="/home/hkl/Cache4Diffusion/flux_simple:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="6,7"

# Smoothing parameters
USE_SMOOTHING="False"
USE_HYBRID_SMOOTHING="False"
SMOOTHING_METHOD="exponential"

N=(0)
O=(0)
F=(50)
alphas=(0)
steps=(50 16 10)

for n in "${N[@]}"; do
    for o in "${O[@]}"; do
        for f in "${F[@]}"; do
            echo "Running inference with N=$n O=$o F=$f ..."

            export FRESH_THRESHOLD="$n"
            export MAX_ORDER="$o"
            export FIRST_ENHANCE="$f"

            for alpha in "${alphas[@]}"; do
                echo "Running inference with dtype=$dtype SMOOTHING_ALPHA=$alpha ..."

                if [ "$alpha" = 0 ]; then
                    export USE_SMOOTHING="False"
                else
                    export USE_SMOOTHING="True"
                    export SMOOTHING_ALPHA="$alpha"
                fi

                # 从 lora_path 提取文件名（无扩展名）作为目录标签
                if [ -n "$lora_path" ]; then
                    lora_name="lora-$(basename "$lora_path" .safetensors)"
                else
                    lora_name="lora-none"
                fi

                for step in "${steps[@]}"; do
                    current_outdir="${base_outdir}/${dtype}/${lora_name}/quant-${quantize:-none}/S${step}/N${n}O${o}F${f}A${alpha}"

                    python taylorseer_flux/batch_infer.py \
                        --model "$local_model_path" \
                        --steps "$step" \
                        --seed 42 \
                        --guidance_scale 7.5 \
                        --outdir "$current_outdir" \
                        --prefix ts_smooth \
                        --prompt_file "$prompt_file" \
                        --dtype "$dtype" \
                        ${quantize:+--quantize "$quantize"} \
                        ${transformer_file:+--transformer_file "$transformer_file"} \
                        ${lora_path:+--lora "$lora_path"} \
                        ${lora_path:+--lora_scale "$lora_scale"}

                    echo "Finished dtype=$dtype alpha=$alpha step=$step. Results saved to $current_outdir"
                done
            done
        done
    done
done