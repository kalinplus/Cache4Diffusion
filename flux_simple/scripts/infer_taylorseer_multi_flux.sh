local_model_path="/mnt/data0/pretrained_models/black-forest-labs/FLUX.1-dev"
model_id="black-forest-labs/FLUX.1-dev"

base_outdir="/home/hkl/Cache4Diffusion/flux_simple/outputs/"
prompt_file="/home/hkl/Cache4Diffusion/assets/prompts/DrawBench200.txt"

# 支持的 dtype: float16, bfloat16, float8, float32
dtype="float8"

cd flux_simple
export CUDA_VISIBLE_DEVICES='0'

# Smoothing parameters
USE_SMOOTHING="False"
USE_HYBRID_SMOOTHING="False"
SMOOTHING_METHOD="exponential"

N=(3 5 6)
O=(0 1 2)
F=(3)
alphas=(0 0.8)

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

                # 为每个 N, O, F, alpha 组合创建独立的输出子目录
                current_outdir="${base_outdir}/${dtype}/N${n}O${o}F${f}A${alpha}"

                python taylorseer_flux/batch_infer.py \
                    --model "$local_model_path" \
                    --steps 50 \
                    --seed 42 \
                    --dtype "$dtype" \
                    --guidance_scale 7.5 \
                    --outdir "$current_outdir" \
                    --prefix ts_smooth \
                    --prompt_file "$prompt_file"

                echo "Finished dtype=$dtype alpha=$alpha. Results saved to $current_outdir"
            done
        done
    done
done