#!/bin/bash
#
# Parallel experiment runner for HunyuanImage TaylorSeer experiments
# Iterates over different smoothing and cache parameters (interval, max_order, first_enhance)
# Automatically distributes experiments across specified GPUs
#
# To explore more cache parameter combinations, modify the lists in parallel_taylorseer_experiments.py:
#   - CACHE_INTERVALS: Cache interval values (default: [5])
#   - CACHE_MAX_ORDERS: Maximum Taylor order values (default: [2])
#   - CACHE_FIRST_ENHANCES: First enhancement step values (default: [3])

PROJECT_ROOT="/home/hkl/Cache4Diffusion"
cd "${PROJECT_ROOT}"

# GPU configuration - modify as needed
GPUS="4,5,6,7"

# Optional: Specify a range of prompts to process
# Uncomment and modify these lines to process only a subset
# START_IDX="--start_idx 0"
# END_IDX="--end_idx 10"
# START_IDX=""
# END_IDX=""

# Optional: Limit parallel experiments (default: use all GPUs)
# MAX_PARALLEL="--max_parallel 4"
MAX_PARALLEL=""

# Optional: Dry run to see what experiments will be run
# DRY_RUN="--dry_run"
DRY_RUN=""

echo "=========================================="
echo "Starting parallel TaylorSeer experiments"
echo "GPUs: $GPUS"
echo "=========================================="

python HunyuanImage-2.1/scripts/parallel_taylorseer_experiments.py \
    --gpus "$GPUS" \
    --project_root "$PROJECT_ROOT" \
    $START_IDX \
    $END_IDX \
    $MAX_PARALLEL \
    $DRY_RUN

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
