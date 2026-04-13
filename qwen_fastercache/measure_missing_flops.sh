#!/usr/bin/env bash
# 补测 start3_interval5 / start3_interval8 的 FLOPs（单张图，快速）
# 结果写入 run_target_settings.log，供 evaluate_fastercache.py 提取

LOG_FILE="run_target_settings.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "========== measure_missing_flops.sh 开始 $(date) =========="

source /apdcephfs_zwfy8/share_304210317/jiachengliu/envs/qwen/bin/activate 2>/dev/null || true

GPUS=0,1,2,3,4,5,6,7
NPROC=8
MASTER_PORT=29700
MODEL_NAME="qwen-image"
MODEL_PATH="/apdcephfs_zwfy8/share_304210317/jiachengliu/pretrained_models/Qwen/Qwen-Image"
# 只用 1 条 prompt 就够，直接 echo 到临时文件
PROMPT_FILE=$(mktemp /tmp/flops_prompt_XXXX.txt)
echo "A beautiful sunset over the ocean with vibrant colors" > "${PROMPT_FILE}"
BASE_DIR="samples/fastercache"

measure_flops() {
    local TAG=$1 START=$2 INTERVAL=$3 ALPHA=$4
    local OUT="${BASE_DIR}/flops_${TAG}"
    mkdir -p "${OUT}"

    echo ""
    echo "=============================================="
    echo " Setting: ${TAG}"
    echo "   [FLOPs]  fc_start_step=${START}  fc_interval=${INTERVAL}  fc_alpha=${ALPHA}"
    echo "=============================================="

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

measure_flops "start3_interval5_alpha0.3"  3 5 0.3
measure_flops "start3_interval8_alpha0.3"  3 8 0.3

rm -f "${PROMPT_FILE}"
echo ""
echo "========== FLOPs 测量完成 $(date) =========="
echo "结果已追加到 ${LOG_FILE}，现在用 evaluate_fastercache.py 提取 ..."

# 直接打印提取结果
source /apdcephfs_zwfy8/share_304210317/jiachengliu/envs/stablediffusion/bin/activate 2>/dev/null || true

IR_MODEL="/apdcephfs_zwfy8/share_304210317/jiachengliu/checkpoint/ImageReward"
FULL_PROMPT="/apdcephfs_cq11/share_300483685/jiachengliu/code/qwen_final/freqca/prompts/DrawBench200.txt"

for TAG in "start3_interval5_alpha0.3" "start3_interval8_alpha0.3"; do
    echo ""
    echo "--- 提取 FLOPs: ${TAG} ---"
    python evaluate_fastercache.py \
        --test_folder   "${BASE_DIR}/${TAG}" \
        --log_file      "${LOG_FILE}" \
        --prompt_file   "${FULL_PROMPT}" \
        --imagereward_model_path "${IR_MODEL}" \
        --output_log    "eval_target_settings.log"
done

echo ""
echo "========================================================"
echo "  更新后的完整报告："
echo "========================================================"
cat eval_target_settings.log
