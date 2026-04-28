#!/bin/bash
set -euo pipefail

# 汇总 VBench 评估结果为最终评分
#
# Usage: bash scripts/run_tabulate.sh [score_dir]
#   score_dir: 包含 vbench/ 子目录的路径（即 launch_calc.sh 的 CKPT_DIR）
#              默认: results/vbench-simulate/N5O1F1A0.8/scores

IMAGE_DIR="/home/hkl/TaylorSeer/TaylorSeer-HunyuanVideo/results/vbench-simulate/N3O1F1A0.8"
SCORE_DIR="$IMAGE_DIR/scores/vbench"

python eval/vbench/tabulate_vbench_scores.py \
    --score_dir "$SCORE_DIR"
