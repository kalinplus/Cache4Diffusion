#!/usr/bin/env bash
# 目标 4 组 FasterCache setting 完整测试
# ─────────────────────────────────────────────────────────────────────────────
# Setting                      ImageReward  FLOPs  Speedup
# start1_interval3_alpha0.3    0.4366       待测   待测      ← 图已有，只跑 FLOPs
# start3_interval5_alpha0.3    待测         待测   待测      ← 需生成图 + FLOPs + IR
# start3_interval8_alpha0.3    待测         待测   待测      ← 需生成图 + FLOPs + IR
# start15_interval2_alpha0.3   0.6804       待测   待测      ← 图已有，只跑 FLOPs
#
# 所有 stdout/stderr 写入 run_target_settings.log（FLOPs 提取依赖此 log）
# ─────────────────────────────────────────────────────────────────────────────

LOG_FILE="run_target_settings.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "========== run_target_settings.sh 开始 $(date) =========="

GPUS=0,1,2,3,4,5,6,7
NPROC=8
MASTER_PORT=29600
MODEL_NAME="qwen-image"
MODEL_PATH="/apdcephfs_zwfy8/share_304210317/jiachengliu/pretrained_models/Qwen/Qwen-Image"
PROMPT_FILE="/apdcephfs_cq11/share_300483685/jiachengliu/code/qwen_final/freqca/prompts/DrawBench200.txt"
BASE_DIR="samples/fastercache"

# ── 阶段一：生成（qwen 环境）────────────────────────────────────────────────
source /apdcephfs_zwfy8/share_304210317/jiachengliu/envs/qwen/bin/activate 2>/dev/null || true

# 辅助函数：生成图片
gen_images() {
    local TAG=$1 START=$2 INTERVAL=$3 ALPHA=$4
    local OUT="${BASE_DIR}/${TAG}"

    echo ""
    echo "=============================================="
    echo " Setting: ${TAG}"
    echo "   fc_start_step=${START}  fc_interval=${INTERVAL}  fc_alpha=${ALPHA}"
    echo "=============================================="

    mkdir -p "${OUT}"
    CUDA_VISIBLE_DEVICES=${GPUS} torchrun \
        --nproc_per_node=${NPROC} --master_port=${MASTER_PORT} \
        sample_ddp.py \
            --model_name        "${MODEL_NAME}" \
            --model_path        "${MODEL_PATH}" \
            --prompt_file       "${PROMPT_FILE}" \
            --output_dir        "${OUT}" \
            --width 1328 --height 1328 --num_steps 50 \
            --guidance_scale 1.0 --true_cfg_scale 1.0 \
            --seed 0 --num_images_per_prompt 1 --batch_size 1 \
            --fc_start_step "${START}" \
            --fc_interval   "${INTERVAL}" \
            --fc_alpha      "${ALPHA}"
    MASTER_PORT=$((MASTER_PORT + 1))
}

# 辅助函数：FLOPs 测量（只跑少量图，快速）
measure_flops() {
    local TAG=$1 START=$2 INTERVAL=$3 ALPHA=$4
    local OUT="${BASE_DIR}/flops_${TAG}"

    echo ""
    echo "=============================================="
    echo " Setting: ${TAG}"
    echo "   [FLOPs mode]  fc_start_step=${START}  fc_interval=${INTERVAL}  fc_alpha=${ALPHA}"
    echo "=============================================="

    mkdir -p "${OUT}"
    CUDA_VISIBLE_DEVICES=${GPUS} torchrun \
        --nproc_per_node=${NPROC} --master_port=${MASTER_PORT} \
        sample_ddp.py \
            --model_name        "${MODEL_NAME}" \
            --model_path        "${MODEL_PATH}" \
            --prompt_file       "${PROMPT_FILE}" \
            --output_dir        "${OUT}" \
            --width 1328 --height 1328 --num_steps 50 \
            --guidance_scale 1.0 --true_cfg_scale 1.0 \
            --seed 0 --num_images_per_prompt 1 --batch_size 1 \
            --fc_start_step "${START}" \
            --fc_interval   "${INTERVAL}" \
            --fc_alpha      "${ALPHA}" \
            --test_FLOPs
    MASTER_PORT=$((MASTER_PORT + 1))
}

# ── 1-A. 生成缺失的两组图片 ─────────────────────────────────────────────────
echo ""
echo ">>> [生成] start3_interval5_alpha0.3"
gen_images "start3_interval5_alpha0.3"  3 5 0.3

echo ""
echo ">>> [生成] start3_interval8_alpha0.3"
gen_images "start3_interval8_alpha0.3"  3 8 0.3

# ── 1-B. FLOPs 测量（全部 4 组）────────────────────────────────────────────
echo ""
echo ">>> [FLOPs] 开始测量全部 4 组 setting …"

measure_flops "start1_interval3_alpha0.3"   1  3 0.3
measure_flops "start3_interval5_alpha0.3"   3  5 0.3
measure_flops "start3_interval8_alpha0.3"   3  8 0.3
measure_flops "start15_interval2_alpha0.3" 15  2 0.3

echo ""
echo "========== 生成 + FLOPs 测量完成 $(date) =========="

# ── 阶段二：ImageReward + 汇总报告（stablediffusion 环境）────────────────────
source /apdcephfs_zwfy8/share_304210317/jiachengliu/envs/stablediffusion/bin/activate

EVAL_LOG="eval_target_settings.log"
IR_MODEL="/apdcephfs_zwfy8/share_304210317/jiachengliu/checkpoint/ImageReward"

SETTINGS=(
    "start1_interval3_alpha0.3"
    "start3_interval5_alpha0.3"
    "start3_interval8_alpha0.3"
    "start15_interval2_alpha0.3"
)

echo ""
echo ">>> [ImageReward + FLOPs 汇总] 开始评测 …"

for TAG in "${SETTINGS[@]}"; do
    FOLDER="${BASE_DIR}/${TAG}"
    if [[ ! -d "${FOLDER}" ]]; then
        echo "[skip] ${TAG} — 目录不存在"
        continue
    fi
    IMG_COUNT=$(find "${FOLDER}" -maxdepth 1 \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
    if [[ "${IMG_COUNT}" -eq 0 ]]; then
        echo "[skip] ${TAG} — 无图像文件"
        continue
    fi

    echo ""
    echo "--- 评测: ${TAG}  (${IMG_COUNT} 张图) ---"
    python evaluate_fastercache.py \
        --test_folder   "${FOLDER}" \
        --log_file      "${LOG_FILE}" \
        --prompt_file   "${PROMPT_FILE}" \
        --imagereward_model_path "${IR_MODEL}" \
        --output_log    "${EVAL_LOG}"
done

echo ""
echo "========================================================"
echo "  FINAL REPORT"
echo "========================================================"
cat "${EVAL_LOG}"
echo ""
echo "Done. 详细日志 → ${LOG_FILE}  |  评测报告 → ${EVAL_LOG}"
