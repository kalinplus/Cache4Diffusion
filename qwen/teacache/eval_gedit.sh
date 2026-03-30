#!/bin/bash

# GEdit 评测脚本 (Qwen-Image-Edit + TeaCache)
# 用于评测 TeaCache 生成的 GEdit 图片质量
# 使用 VIEScore (GPT4o 或 Qwen25VL) 作为评估 backbone

set -euo pipefail

cd qwen/teacache

export CUDA_VISIBLE_DEVICES=6

# 评测参数（请按需修改）
INSTRUCTION_LANGUAGE="en"                          # 评测语言: en / cn / all
TASK_TYPE="all"                                    # 评测任务类型: all / background_change / color_alter 等
BACKBONE="qwen25vl"                                # 评估 backbone: gpt4o / qwen25vl

# 评测配置组合（与 sample_gedit.sh 对齐）
REL_L1_THRESH=(1.0 1.2 1.4)

# 本地模型路径
export GEDIT_DATASET_PATH="/mnt/data0/datasets/stepfun-ai/GEdit-Bench"
export QWEN25VL_MODEL_PATH="/mnt/data0/Qwen/Qwen2.5-VL-72B-Instruct-AWQ"

for rel_l1_thresh in "${REL_L1_THRESH[@]}"; do
    SAVE_DIR="/home/hkl/Cache4Diffusion/samples/GEdit/qwen-image-edit/teacache/R${rel_l1_thresh}"

    # 检查 fullset 是否存在且有图片
    if [ ! -d "${SAVE_DIR}/fullset" ]; then
        echo "=========================================="
        echo "Skipping (no fullset): R${rel_l1_thresh}"
        echo "=========================================="
        continue
    fi

    echo "=========================================="
    echo "Evaluating: R${rel_l1_thresh}"
    echo "=========================================="

    python /home/hkl/Cache4Diffusion/qwen/teacache/evaluate_gedit.py \
        --save_dir "${SAVE_DIR}" \
        --instruction_language "${INSTRUCTION_LANGUAGE}" \
        --task_type "${TASK_TYPE}" \
        --backbone "${BACKBONE}"
done

# 参数说明：
# --save_dir: 评测图片目录（格式为 samples/GEdit/qwen-image-edit/teacache/R{rel_l1_thresh}）
# --instruction_language: 评测语言（en/cn/all）
# --task_type: 评测任务类型，支持以下类型：
#   all, background_change, color_alter, material_alter, motion_change,
#   ps_human, style_change, subject-add, subject-remove, subject-replace,
#   text_change, tone_transfer
# --backbone: VIEScore 评估 backbone（gpt4o 或 qwen25vl）
#
# 输出目录结构：
#   {SAVE_DIR}/score/{task_type}.csv  - 每种任务类型的评分结果
#   {SAVE_DIR}/score/scores.csv       - 汇总评分结果
