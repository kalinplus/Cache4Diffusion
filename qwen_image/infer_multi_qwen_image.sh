local_model_path="/data/public/models/Qwen/Qwen-Image"
model_id="Qwen/Qwen-Image"
# 基础输出目录
base_outdir="/data/huangkailin-20250908/Cache4Diffusion/qwen_image/outputs/smooth/exp"
prompt_file="/data/huangkailin-20250908/Cache4Diffusion/assets/prompts/DrawBench200.txt"

PROJECT_ROOT="/data/huangkailin-20250908/Cache4Diffusion"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES='7'

# 设置固定的平滑参数
export USE_SMOOTHING="True"
export USE_HYBRID_SMOOTHING="False"
export SMOOTHING_METHOD="exponential"

# 定义要遍历的 alpha 列表
alphas=(0.75 0.8 0.85 0.9 0.95)

for alpha in "${alphas[@]}"; do
    echo "Running inference with SMOOTHING_ALPHA=$alpha ..."
    
    # 设置当前循环的 alpha 环境变量
    export SMOOTHING_ALPHA="$alpha"
    
    # 为每个 alpha 创建独立的输出子目录
    current_outdir="${base_outdir}/${alpha}"
    
    python qwen_image/taylorseer_qwen_image/batch_infer.py \
        --model "$local_model_path" \
        --steps 50 \
        --seed 42 \
        --dtype bfloat16 \
        --outdir "$current_outdir" \
        --prefix ts_smooth \
        --true_cfg_scale 7.5 \
        --prompt_file "$prompt_file" \
        --use_taylor
        
    echo "Finished alpha=$alpha. Results saved to $current_outdir"
done