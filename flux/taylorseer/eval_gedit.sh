#!/bin/bash

# GEdit 评测脚本
# 用于评测 TaylorSeer 生成的 GEdit 图片质量
# 使用 VIEScore (GPT4o 或 Qwen25VL) 作为评估 backbone

set -euo pipefail

cd flux/taylorseer

export CUDA_VISIBLE_DEVICES=2

# # 项目根目录（用于 PYTHONPATH）
# PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# 评测参数（请按需修改）
INSTRUCTION_LANGUAGE="en"                          # 评测语言: en / cn / all
TASK_TYPE="all"                                    # 评测任务类型: all / background_change / color_alter 等
BACKBONE="qwen25vl"                                # 评估 backbone: gpt4o / qwen25vl

# 评测配置组合
INTERVALS=(4 5 6)
MAX_ORDERS=(1)
FIRST_ENHANCES=(3)
ALPHAS=(0 0.8)

# 本地模型路径（可选）
# GEDIT_DATASET_PATH: GEdit-Bench 数据集本地路径
# QWEN25VL_MODEL_PATH: Qwen2.5-VL-72B-Instruct-AWQ 本地模型路径
export GEDIT_DATASET_PATH="/mnt/data0/datasets/stepfun-ai/GEdit-Bench"
export QWEN25VL_MODEL_PATH="/mnt/data0/Qwen/Qwen2.5-VL-72B-Instruct-AWQ"

for interval in "${INTERVALS[@]}"; do
    for max_order in "${MAX_ORDERS[@]}"; do
        for first_enhance in "${FIRST_ENHANCES[@]}"; do
            for alpha in "${ALPHAS[@]}"; do
                SAVE_DIR="/home/hkl/Cache4Diffusion/samples/GEdit/taylorseer/N${interval}O${max_order}F${first_enhance}Alpha${alpha}"

                echo "=========================================="
                echo "Evaluating: N${interval}O${max_order}F${first_enhance}Alpha${alpha}"
                echo "=========================================="

                python /home/hkl/Cache4Diffusion/flux/taylorseer/evaluate_gedit.py \
                    --save_dir "${SAVE_DIR}" \
                    --instruction_language "${INSTRUCTION_LANGUAGE}" \
                    --task_type "${TASK_TYPE}" \
                    --backbone "${BACKBONE}"
            done
        done
    done
done

# 参数说明：
# --save_dir: 评测图片目录（由脚本自动生成，格式为 samples/GEdit/taylorseer/N${interval}O${max_order}F${first_enhance}Alpha${alpha}）
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