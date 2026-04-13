#!/usr/bin/env bash
# 从已有 log 重新提取 FLOPs，更新 eval_target_settings.log
source /apdcephfs_zwfy8/share_304210317/jiachengliu/envs/stablediffusion/bin/activate

LOG_FILE="run_target_settings.log"
EVAL_LOG="eval_target_settings.log"
IR_MODEL="/apdcephfs_zwfy8/share_304210317/jiachengliu/checkpoint/ImageReward"
PROMPT_FILE="/apdcephfs_cq11/share_300483685/jiachengliu/code/qwen_final/freqca/prompts/DrawBench200.txt"
BASE_DIR="samples/fastercache"

rm -f "${EVAL_LOG}"

for TAG in \
    "start1_interval3_alpha0.3" \
    "start3_interval5_alpha0.3" \
    "start3_interval8_alpha0.3" \
    "start15_interval2_alpha0.3"; do

    python evaluate_fastercache.py \
        --test_folder "${BASE_DIR}/${TAG}" \
        --log_file    "${LOG_FILE}" \
        --prompt_file "${PROMPT_FILE}" \
        --imagereward_model_path "${IR_MODEL}" \
        --output_log  "${EVAL_LOG}"
done

echo ""
echo "========================================================"
echo "  完整报告："
echo "========================================================"
cat "${EVAL_LOG}"
