#!/bin/bash
# 批量评测所有 FasterCache setting
# 遍历 samples/fastercache/ 下的每个子文件夹，逐一计算 ImageReward 并从 run.log 提取 FLOPs。
# 结果汇总到 eval_report.log，屏幕实时显示。
#
# 用法：
#   bash evaluate_all.sh                    # 评测所有 setting
#   bash evaluate_all.sh start1_interval8   # 只评测名字包含该字符串的 setting

source /apdcephfs_zwfy8/share_304210317/jiachengliu/envs/stablediffusion/bin/activate


SAMPLES_DIR="samples/fastercache"
LOG_FILE="run.log"
PROMPT_FILE="/apdcephfs_cq11/share_300483685/jiachengliu/code/qwen_final/freqca/prompts/DrawBench200.txt"
IR_MODEL="/apdcephfs_zwfy8/share_304210317/jiachengliu/checkpoint/ImageReward"
EVAL_LOG="eval_report.log"
FILTER="${1:-}"   # 可选：只评测名字含该字符串的 setting

# 清空旧报告（首次运行时）
rm -f "${EVAL_LOG}"

SETTINGS=(
    "start15_interval2_alpha0.3"
    "start1_interval3_alpha0.3"
    "start1_interval8_alpha0.3"
)

for TAG in "${SETTINGS[@]}"; do
    FOLDER="${SAMPLES_DIR}/${TAG}"

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
    echo ">>> 评测: ${TAG}  (${IMG_COUNT} 张图)"
    python evaluate_fastercache.py \
        --test_folder   "${FOLDER}" \
        --log_file      "${LOG_FILE}" \
        --prompt_file   "${PROMPT_FILE}" \
        --imagereward_model_path "${IR_MODEL}" \
        --output_log    "${EVAL_LOG}"
done

echo ""
echo "========================================"
echo "  全部评测完成，汇总报告："
echo "========================================"
cat "${EVAL_LOG}"
