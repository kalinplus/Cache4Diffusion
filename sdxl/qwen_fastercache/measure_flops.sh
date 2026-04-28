#!/bin/bash
# FasterCache 加速比扫描脚本
#
# 每组 setting 只跑 1 条 prompt、1 张图，纯 FLOPs 统计，不保存结果图。
# 加速比汇总写入 LOG_FILE，方便对比。
#
# 用法：
#   bash measure_flops.sh
#   cat flops_report.log

source /apdcephfs_zwfy8/share_304210317/jiachengliu/envs/qwen/bin/activate

LOG_FILE="flops_report.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "========== measure_flops.sh 开始 $(date) =========="

NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-29500}

MODEL_NAME="qwen-image"
MODEL_PATH="/apdcephfs_zwfy8/share_304210317/jiachengliu/pretrained_models/Qwen/Qwen-Image"
PROMPT_FILE="/apdcephfs_cq11/share_300483685/jiachengliu/code/qwen_final/freqca/prompts/DrawBench200.txt"

LOG_FILE="flops_report.log"
TMP_OUT="samples/flops_tmp"
mkdir -p "${TMP_OUT}"

# 只取第 1 条 prompt，避免跑全部 200 条
SINGLE_PROMPT=$(mktemp /tmp/fc_prompt_XXXX.txt)
head -1 "${PROMPT_FILE}" > "${SINGLE_PROMPT}"

# 初始化 log
echo "FasterCache 加速比测量报告 — $(date)" >  "${LOG_FILE}"
echo "模型: ${MODEL_NAME}  步数: 50  分辨率: 1328x1328" >> "${LOG_FILE}"
echo "基线 FLOPs（无缓存，50步）: 12917.56 T" >> "${LOG_FILE}"
echo "---------------------------------------------------" >> "${LOG_FILE}"
printf "%-35s %12s %12s\n" "Setting" "FLOPs(T)" "Speedup" >> "${LOG_FILE}"
echo "---------------------------------------------------" >> "${LOG_FILE}"

measure() {
    local TAG=$1 START=$2 INTERVAL=$3 ALPHA=$4

    echo "[running] ${TAG}  (start=${START} interval=${INTERVAL} alpha=${ALPHA})"

    # 运行并捕获输出
    RAW=$(CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port="${MASTER_PORT}" \
        sample_ddp.py \
            --model_name  "${MODEL_NAME}" \
            --model_path  "${MODEL_PATH}" \
            --prompt_file "${SINGLE_PROMPT}" \
            --output_dir  "${TMP_OUT}" \
            --width 1328 --height 1328 \
            --num_steps 50 \
            --guidance_scale 1.0 --true_cfg_scale 1.0 \
            --seed 0 \
            --num_images_per_prompt 1 \
            --batch_size 1 \
            --fc_start_step  "${START}" \
            --fc_interval    "${INTERVAL}" \
            --fc_alpha       "${ALPHA}" \
            --test_FLOPs 2>&1)

    # 从输出里提取数字
    FLOPS=$(echo "${RAW}"   | grep "Total FLOPs" | grep -oP '[0-9]+\.[0-9]+' | head -1)
    SPEEDUP=$(echo "${RAW}" | grep "Speedup"     | grep -oP '[0-9]+\.[0-9]+x' | head -1)

    # 写入 log
    printf "%-35s %12s %12s\n" "${TAG}" "${FLOPS}T" "${SPEEDUP}" >> "${LOG_FILE}"
    echo "  → FLOPs=${FLOPS}T  Speedup=${SPEEDUP}"

    MASTER_PORT=$((MASTER_PORT + 1))
}


# ── start=5，interval=6（对比组） ───────────────────────────
# measure "start3_interval6_alpha0.3"     3  6  0.3

# ── start=5，interval=6（宽松 vs 激进对比） ─────────────────
measure "start3_interval12_alpha0.3"    1  12  0.3
measure "start3_interval8_alpha0.3"    1  8  0.3
echo "---------------------------------------------------" >> "${LOG_FILE}"
echo "" >> "${LOG_FILE}"

rm -f "${SINGLE_PROMPT}"

echo ""
echo "======================================="
echo "  测量完成 $(date)"
echo "  结果已写入 ${LOG_FILE}"
echo "======================================="
