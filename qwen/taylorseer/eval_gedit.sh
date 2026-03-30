#!/bin/bash

# GEdit 评测脚本 (Qwen-Image-Edit + TaylorSeer)
# 用于评测 TaylorSeer 生成的 GEdit 图片质量
# 使用 VIEScore (GPT4o 或 Qwen25VL) 作为评估 backbone

set -euo pipefail

cd qwen/taylorseer

export CUDA_VISIBLE_DEVICES=7

# 评测参数（请按需修改）
INSTRUCTION_LANGUAGE="en"                          # 评测语言: en / cn / all
TASK_TYPE="all"                                    # 评测任务类型: all / background_change / color_alter 等
BACKBONE="qwen25vl"                                # 评估 backbone: gpt4o / qwen25vl

# 评测配置组合
INTERVALS=(5 6 9)
MAX_ORDERS=(1 2)
FIRST_ENHANCES=(3)
ALPHAS=(0 0.8)

# OOM 配置：O=2 + α=0.8 在单卡 H20 上无法跑完，跳过
SKIP_CONFIGS=("N5O2F3A0.8" "N6O2F3A0.8" "N9O2F3A0.8")

# 本地模型路径
export GEDIT_DATASET_PATH="/mnt/data0/datasets/stepfun-ai/GEdit-Bench"
export QWEN25VL_MODEL_PATH="/mnt/data0/Qwen/Qwen2.5-VL-72B-Instruct-AWQ"

for interval in "${INTERVALS[@]}"; do
    for max_order in "${MAX_ORDERS[@]}"; do
        for first_enhance in "${FIRST_ENHANCES[@]}"; do
            for alpha in "${ALPHAS[@]}"; do
                CONFIG_NAME="N${interval}O${max_order}F${first_enhance}A${alpha}"

                # 跳过 OOM 配置
                skip=false
                for skip_cfg in "${SKIP_CONFIGS[@]}"; do
                    if [ "$CONFIG_NAME" = "$skip_cfg" ]; then
                        skip=true
                        break
                    fi
                done
                if [ "$skip" = true ]; then
                    echo "=========================================="
                    echo "Skipping (OOM): ${CONFIG_NAME}"
                    echo "=========================================="
                    continue
                fi

                SAVE_DIR="/home/hkl/Cache4Diffusion/samples/GEdit/qwen-image-edit/taylorseer/${CONFIG_NAME}"

                # 检查 fullset 是否存在且有图片
                if [ ! -d "${SAVE_DIR}/fullset" ]; then
                    echo "=========================================="
                    echo "Skipping (no fullset): ${CONFIG_NAME}"
                    echo "=========================================="
                    continue
                fi

                echo "=========================================="
                echo "Evaluating: ${CONFIG_NAME}"
                echo "=========================================="

                python /home/hkl/Cache4Diffusion/qwen/taylorseer/evaluate_gedit.py \
                    --save_dir "${SAVE_DIR}" \
                    --instruction_language "${INSTRUCTION_LANGUAGE}" \
                    --task_type "${TASK_TYPE}" \
                    --backbone "${BACKBONE}"
            done
        done
    done
done

# 参数说明：
# --save_dir: 评测图片目录（格式为 samples/GEdit/qwen-image-edit/taylorseer/N{N}O{O}F{F}A{alpha}）
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
