#!/bin/bash
# =============================================================================
# SLURM BATCH JOB TEMPLATE
# =============================================================================
# Usage: sbatch run_flux_taylorseer.sbatch
# Or with overrides:
#   sbatch --time=01:00:00 run_flux_taylorseer.sbatch
#   sbatch --array=0-4 run_flux_taylorseer.sbatch  # Array jobs
# =============================================================================

# Job name (shows in squeue)
#SBATCH --job-name=my_job

# Output files (%j = job ID)
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err

# Time limit (REQUIRED for GPU jobs!)
# Format: hh:mm:ss
#SBATCH --time=02:00:00

# Partition selection (CRITICAL!)
# - a100_global: data on /mnt (shared)
# - a100_local: data on /data (local)
#SBATCH --partition=a100_global

# GPU allocation
# Option A: Let Slurm select (recommended for single GPU)
#SBATCH --gres=gpu
# Option B: Request specific number
#SBATCH --gres=gpu:2

# CPU cores
#SBATCH --cpus-per-task=16

# Memory (optional, usually not needed to specify)
#SBATCH --mem=64G

# Email notifications (optional)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your@email.com

# =============================================================================
# JOB ENVIRONMENT
# =============================================================================
# Working directory
cd /data/huangkailin-20250908/Cache4Diffusion/flux/taylorseer

# NOTE: Do NOT set CUDA_VISIBLE_DEVICES!
# Slurm allocates GPUs, don't override with specific IDs
# Remove any "export CUDA_VISIBLE_DEVICES=0,1" from your original script

# =============================================================================
# YOUR COMMAND HERE
# =============================================================================
# Direct command:
# python some_script.py --arg1 value1

# Or call another script:
# bash run_flux_taylorseer.sh --num_steps 50

echo "Job $SLURM_JOB_ID started on $(hostname)"
echo " GPUs allocated: $CUDA_VISIBLE_DEVICES"
