local_model_path="/data/public/models/FLUX.1-dev"
model_id="black-forest-labs/FLUX.1-dev"
# 基础输出目录
base_outdir="/data/huangkailin-20250908/Cache4Diffusion/flux/outputs/smooth/db200/exp"
prompt_file="/data/huangkailin-20250908/Cache4Diffusion/assets/prompts/DrawBench200.txt"

cd flux
export CUDA_VISIBLE_DEVICES='0'

# 设置固定的平滑参数
export USE_SMOOTHING="True"
export USE_HYBRID_SMOOTHING="False"
export SMOOTHING_METHOD="exponential"

# 定义要遍历的 alpha 列表
alphas=(0.75 0.775 0.8 0.825 0.85 0.875 0.9 0.925 0.95)

for alpha in "${alphas[@]}"; do
    echo "Running inference with SMOOTHING_ALPHA=$alpha ..."
    
    # 设置当前循环的 alpha 环境变量
    export SMOOTHING_ALPHA="$alpha"
    
    # 为每个 alpha 创建独立的输出子目录
    current_outdir="${base_outdir}/${alpha}"
    
    python taylorseer_flux/batch_infer.py \
        --model "$local_model_path" \
        --steps 50 \
        --seed 42 \
        --dtype float16 \
        --guidance_scale 7.5 \
        --outdir "$current_outdir" \
        --prefix ts_smooth \
        --prompt_file "$prompt_file"
        
    echo "Finished alpha=$alpha. Results saved to $current_outdir"
done