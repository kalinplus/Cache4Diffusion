#!/bin/bash
# Qwen image generation with FasterCache — 多组 setting 对比测试
#
# FasterCache schedule 调优参考（50步）：
#   保守 : fc_start_step=20, fc_interval=2, fc_alpha=0.1   (~30% speedup)
#   均衡 : fc_start_step=15, fc_interval=2, fc_alpha=0.3   (~40% speedup)
#   激进 : fc_start_step=10, fc_interval=3, fc_alpha=0.3   (~55% speedup)

source /apdcephfs_zwfy8/share_304210317/jiachengliu/envs/qwen/bin/activate
set -e

LOG_FILE="run.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "========== run.sh 开始 $(date) =========="

NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-29500}

MODEL_NAME="qwen-image"
MODEL_PATH="/apdcephfs_zwfy8/share_304210317/jiachengliu/pretrained_models/Qwen/Qwen-Image"
PROMPT_FILE="/apdcephfs_cq11/share_300483685/jiachengliu/code/qwen_final/freqca/prompts/DrawBench200.txt"
BASE_DIR="samples/fastercache"

run_setting() {
    local TAG=$1
    local START=$2
    local INTERVAL=$3
    local ALPHA=$4
    local OUT="${BASE_DIR}/${TAG}"

    echo "=============================================="
    echo " Setting: ${TAG}"
    echo "   fc_start_step=${START}  fc_interval=${INTERVAL}  fc_alpha=${ALPHA}"
    echo "=============================================="

    mkdir -p "${OUT}"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port="${MASTER_PORT}" \
        sample_ddp.py \
            --model_name        "${MODEL_NAME}" \
            --model_path        "${MODEL_PATH}" \
            --prompt_file       "${PROMPT_FILE}" \
            --output_dir        "${OUT}" \
            --width             1328 \
            --height            1328 \
            --num_steps         50 \
            --guidance_scale    1.0 \
            --true_cfg_scale    1.0 \
            --seed              0 \
            --num_images_per_prompt 1 \
            --batch_size        1 \
            --fc_start_step     "${START}" \
            --fc_interval       "${INTERVAL}" \
            --fc_alpha          "${ALPHA}"

    # 每次 torchrun 结束后端口 +1，避免重复使用同一端口
    MASTER_PORT=$((MASTER_PORT + 1))
}


# 均衡：推荐默认配置
run_setting "start15_interval2_alpha0.3"      15 2 0.3

# 激进：更大幅度跳步，速度最快，质量略降
run_setting "start1_interval3_alpha0.3"    1 5 0.3
run_setting "start1_interval8_alpha0.3"    1 8 0.3


# ── FLOPs 加速比测量（只跑 1 张图，快） ───────────────────
# 用 calflops 统计每步实际 FLOPs，对比无缓存基线（12917.56 T）
echo "=============================================="
echo " FLOPs 测量：3 组 setting 加速比对比"
echo "=============================================="

measure_flops() {
    local TAG=$1; local START=$2; local INTERVAL=$3; local ALPHA=$4
    local OUT="${BASE_DIR}/flops_${TAG}"
    mkdir -p "${OUT}"
    echo "--- FLOPs: ${TAG} ---"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port="${MASTER_PORT}" \
        sample_ddp.py \
            --model_name   "${MODEL_NAME}" \
            --model_path   "${MODEL_PATH}" \
            --prompt_file  "${PROMPT_FILE}" \
            --output_dir   "${OUT}" \
            --width 1328 --height 1328 --num_steps 50 \
            --guidance_scale 1.0 --true_cfg_scale 1.0 \
            --seed 0 --num_images_per_prompt 1 --batch_size 1 \
            --fc_start_step "${START}" \
            --fc_interval   "${INTERVAL}" \
            --fc_alpha      "${ALPHA}" \
            --test_FLOPs
    MASTER_PORT=$((MASTER_PORT + 1))
}

measure_flops "conservative" 15 2 0.3
measure_flops "balanced"     1 5 0.3
measure_flops "aggressive"   1 8 0.3
