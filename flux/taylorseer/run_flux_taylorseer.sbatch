#!/bin/bash
#SBATCH --job-name=flux_taylorseer
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err
#SBATCH --time=01:00:00              # Adjust as needed
#SBATCH --partition=a100_global      # Data on /mnt -> use global partition
#SBATCH --gres=gpu                  # Let Slurm select GPU (don't specify :N)
#SBATCH --cpus-per-task=16

# ============================================================================
# Load environment
# ============================================================================
cd /data/huangkailin-20250908/Cache4Diffusion/flux/taylorseer

# NOTE: Do NOT set CUDA_VISIBLE_DEVICES - let Slurm manage GPU allocation
# The script will use all visible GPUs by default

# ============================================================================
# Parameters (customize these)
# ============================================================================
export MODEL_NAME="${MODEL_NAME:-flux-dev}"
export NUM_STEPS="${NUM_STEPS:-50}"
export GUIDANCE="${GUIDANCE:-3.5}"
export SEED="${SEED:-42}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
export INTERVAL="${INTERVAL:-4}"
export MAX_ORDER="${MAX_ORDER:-0}"
export USE_SMOOTHING="${USE_SMOOTHING:-False}"
export SMOOTHING_METHOD="${SMOOTHING_METHOD:-exponential}"
export SMOOTHING_ALPHA="${SMOOTHING_ALPHA:-0.8}"

# ============================================================================
# Run
# ============================================================================
echo "=============================================="
echo "SLURM Job: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=============================================="

bash run_flux_taylorseer.sh \
    --num_steps "${NUM_STEPS}" \
    --guidance "${GUIDANCE}" \
    --seed "${SEED}" \
    --output_dir "${OUTPUT_DIR}" \
    --interval "${INTERVAL}" \
    --max_order "${MAX_ORDER}" \
    --use_smoothing "${USE_SMOOTHING}" \
    --smoothing_method "${SMOOTHING_METHOD}" \
    --smoothing_alpha "${SMOOTHING_ALPHA}"

echo "Done! Job $SLURM_JOB_ID completed."
