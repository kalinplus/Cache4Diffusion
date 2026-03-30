#!/bin/bash
# NOTE: 未适配具体代码
# GEdit Benchmark 评测脚本
# 用于评测 teacache 生成的 GEdit 图片质量
# 使用 VIEScore (GPT4o 或 Qwen25VL) 作为评估 backbone

set -euo pipefail

cd /home/hkl/Cache4Diffusion/flux/teacache

export CUDA_VISIBLE_DEVICES=0

# 评测参数
INSTRUCTION_LANGUAGE="en"                          # 评测语言: en / cn / all
TASK_TYPE="all"                                    # 评测任务类型: all / background_change / color_alter 等
BACKBONE="qwen25vl"                                # 评估 backbone: gpt4o / qwen25vl

# TeaCache 参数组合（与 sample_gedit.sh 保持一致）
REL_L1_THRESH=(0.8 1.0 1.2)

# 数据集路径（可选）
export GEDIT_DATASET_PATH="/mnt/data0/datasets/stepfun-ai/GEdit-Bench"
export QWEN25VL_MODEL_PATH="/mnt/data0/Qwen/Qwen2.5-VL-72B-Instruct-AWQ"

for rel_l1_thresh in "${REL_L1_THRESH[@]}"; do
    SAVE_DIR="/home/hkl/Cache4Diffusion/samples/GEdit/flux-kontext/teacache/R${rel_l1_thresh}"

    echo "=========================================="
    echo "Evaluating: R${rel_l1_thresh}"
    echo "=========================================="

    python evaluate_gedit.py \
        --save_dir "${SAVE_DIR}" \
        --instruction_language "${INSTRUCTION_LANGUAGE}" \
        --task_type "${TASK_TYPE}" \
        --backbone "${BACKBONE}"
done

# 参数说明：
# --save_dir: 评测图片目录（由 sample_gedit.sh 脚本生成）
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
