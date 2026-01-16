#!/usr/bin/env python3
"""
Parallel experiment runner for HunyuanImage TaylorSeer with different smoothing parameters.
Runs experiments in parallel across multiple GPUs using multiprocessing.
"""
import subprocess
import argparse
import os
import sys
from multiprocessing import Process, Queue
from itertools import product

# Experiment configuration
BASE_CONFIG = {
    'model_path': '/mnt/data0/pretrained_models/tencent/HunyuanImage-2.1',
    'model_name': 'hunyuanimage-v2.1',
    'prompt_file': 'assets/prompts/DrawBench200.txt',
    'seed': 649151,
    'width': 2048,
    'height': 2048,
    'shift': 5,
    'guidance_scale': 3.5,
    'use_reprompt': True,
    'use_taylorseer_lite': True,
    'use_refiner': False,
}

# Base output directory
BASE_OUTDIR = '/home/hkl/Cache4Diffusion/outputs/smooth/exp'

# Define experiment grid
# Each experiment config includes smoothing and cache parameters
EXPERIMENTS = []

# Cache parameter combinations to explore
CACHE_INTERVALS = [4, 5, 6]  # Can add more: [3, 4, 5, 6]
CACHE_MAX_ORDERS = [1, 2]  # Can add more: [1, 2]
CACHE_FIRST_ENHANCES = [3]  # Can add more: [2, 3, 4]

# Smoothing parameters
SMOOTHING_ALPHAS = [0.75, 0.8, 0.85, 0.9, 0.95]

# Experiment 1: No smoothing (baseline) - commented out, uncomment if needed
for interval, max_order, first_enhance in product(CACHE_INTERVALS, CACHE_MAX_ORDERS, CACHE_FIRST_ENHANCES):
    EXPERIMENTS.append({
        'use_smoothing': False,  # 没有开平滑，后面两个参数也没用
        'smoothing_method': 'exponential',
        'smoothing_alpha': 0.8,
        'cache_interval': interval,
        'max_order': max_order,
        'first_enhance': first_enhance,
        'label': f'no_smooth_N{interval}O{max_order}F{first_enhance}',
        'outdir_suffix': f'naive_ts/N{interval}O{max_order}F{first_enhance}'
    })

# Experiment 2: Exponential smoothing with different alphas and cache parameters
for interval, max_order, first_enhance in product(CACHE_INTERVALS, CACHE_MAX_ORDERS, CACHE_FIRST_ENHANCES):
    for alpha in SMOOTHING_ALPHAS:
        EXPERIMENTS.append({
            'use_smoothing': True,
            'smoothing_method': 'exponential',
            'smoothing_alpha': alpha,
            'cache_interval': interval,
            'max_order': max_order,
            'first_enhance': first_enhance,
            'label': f'exp_N{interval}O{max_order}F{first_enhance}_alpha_{alpha}',
            'outdir_suffix': f'exponential/N{interval}O{max_order}F{first_enhance}/{alpha}'
        })

# Experiment 3: Moving average smoothing with different cache parameters
for interval, max_order, first_enhance in product(CACHE_INTERVALS, CACHE_MAX_ORDERS, CACHE_FIRST_ENHANCES):
    EXPERIMENTS.append({
        'use_smoothing': True,
        'smoothing_method': 'moving_average',
        'smoothing_alpha': 0.8,  # Different meaning for moving average
        'cache_interval': interval,
        'max_order': max_order,
        'first_enhance': first_enhance,
        'label': f'ma_N{interval}O{max_order}F{first_enhance}_alpha_0.8',
        'outdir_suffix': f'moving_average/N{interval}O{max_order}F{first_enhance}/0.8'
    })


def run_single_experiment(exp_config, gpu_id, project_root, result_queue):
    """Run a single experiment on specified GPU (runs in separate process)."""
    try:
        # Build output directory (BASE_OUTDIR is now absolute path)
        outdir = os.path.join(BASE_OUTDIR, exp_config['outdir_suffix'])
        os.makedirs(outdir, exist_ok=True)

        # Build command
        python_file = os.path.join(project_root, 'HunyuanImage-2.1', 'run_hyimage_taylorseer_lite_batch.py')

        cmd = [
            sys.executable, python_file,
            '--prompt_file', exp_config.get('prompt_file', BASE_CONFIG['prompt_file']),
            '--model_name', exp_config.get('model_name', BASE_CONFIG['model_name']),
            '--seed', str(exp_config.get('seed', BASE_CONFIG['seed'])),
            '--width', str(exp_config.get('width', BASE_CONFIG['width'])),
            '--height', str(exp_config.get('height', BASE_CONFIG['height'])),
            '--shift', str(exp_config.get('shift', BASE_CONFIG['shift'])),
            '--guidance_scale', str(exp_config.get('guidance_scale', BASE_CONFIG['guidance_scale'])),
            '--outdir', outdir,
            '--prefix', f"TS_{exp_config['label']}",
        ]

        # Add optional flags
        if exp_config.get('use_reprompt', BASE_CONFIG['use_reprompt']):
            cmd.append('--use_reprompt')
        if exp_config.get('use_taylorseer_lite', BASE_CONFIG['use_taylorseer_lite']):
            cmd.append('--use_taylorseer_lite')
        if exp_config.get('use_refiner', BASE_CONFIG['use_refiner']):
            cmd.append('--use_refiner')

        # Add batch range if specified
        if 'start_idx' in exp_config:
            cmd.extend(['--start_idx', str(exp_config['start_idx'])])
        if 'end_idx' in exp_config:
            cmd.extend(['--end_idx', str(exp_config['end_idx'])])

        # Add smoothing parameters
        if exp_config['use_smoothing']:
            cmd.append('--use_smoothing')
            cmd.extend(['--smoothing_method', exp_config['smoothing_method']])
            cmd.extend(['--smoothing_alpha', str(exp_config['smoothing_alpha'])])

        # Set cache environment variables
        cache_env_vars = {
            'TS_CACHE_INTERVAL': str(exp_config.get('cache_interval', 5)),
            'TS_MAX_ORDER': str(exp_config.get('max_order', 2)),
            'TS_FIRST_ENHANCE': str(exp_config.get('first_enhance', 3)),
        }

        # Set CUDA_VISIBLE_DEVICES BEFORE any CUDA initialization
        # This must be set in the environment that Python starts in
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        os.environ['HUNYUANIMAGE_V2_1_MODEL_ROOT'] = exp_config.get('model_path', BASE_CONFIG['model_path'])

        # Set cache configuration environment variables
        for key, value in cache_env_vars.items():
            os.environ[key] = value

        print(f"\n{'='*60}")
        print(f"Starting experiment: {exp_config['label']}")
        print(f"GPU: {gpu_id} (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")
        print(f"Output directory: {outdir}")
        print(f"Smoothing: {exp_config['use_smoothing']}, method={exp_config['smoothing_method']}, alpha={exp_config['smoothing_alpha']}")
        print(f"Cache: interval={cache_env_vars['TS_CACHE_INTERVAL']}, max_order={cache_env_vars['TS_MAX_ORDER']}, first_enhance={cache_env_vars['TS_FIRST_ENHANCE']}")
        print(f"{'='*60}\n")

        # Run the command
        result = subprocess.run(cmd, cwd=project_root)

        if result.returncode == 0:
            print(f"\n[SUCCESS] Experiment '{exp_config['label']}' on GPU {gpu_id} completed!")
            result_queue.put(('success', exp_config['label'], gpu_id))
        else:
            print(f"\n[FAILED] Experiment '{exp_config['label']}' on GPU {gpu_id} failed with code {result.returncode}")
            result_queue.put(('failed', exp_config['label'], gpu_id))

        return result.returncode == 0

    except Exception as e:
        print(f"\n[ERROR] Experiment '{exp_config.get('label', 'unknown')}' on GPU {gpu_id} crashed: {e}")
        result_queue.put(('error', exp_config.get('label', 'unknown'), gpu_id))
        return False


def main():
    parser = argparse.ArgumentParser(description='Run parallel TaylorSeer smoothing experiments')
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7',
                        help='Comma-separated list of GPUs to use (default: 0-7)')
    parser.add_argument('--project_root', type=str, default='/home/hkl/Cache4Diffusion',
                        help='Project root directory')
    parser.add_argument('--max_parallel', type=int, default=None,
                        help='Maximum number of parallel experiments (default: use all GPUs)')
    parser.add_argument('--start_idx', type=int, default=None,
                        help='Start index for batch processing')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='End index for batch processing')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print commands without executing')

    args = parser.parse_args()

    # Parse GPU list
    gpu_list = [int(g.strip()) for g in args.gpus.split(',')]
    num_gpus = len(gpu_list)
    max_parallel = args.max_parallel if args.max_parallel else num_gpus
    max_parallel = min(max_parallel, num_gpus)

    print(f"GPUs available: {gpu_list}")
    print(f"Max parallel experiments: {max_parallel}")
    print(f"Total experiments: {len(EXPERIMENTS)}")

    # Add batch range to config if specified
    if args.start_idx is not None or args.end_idx is not None:
        for exp in EXPERIMENTS:
            if args.start_idx is not None:
                exp['start_idx'] = args.start_idx
            if args.end_idx is not None:
                exp['end_idx'] = args.end_idx

    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
        for i, exp in enumerate(EXPERIMENTS):
            gpu_id = gpu_list[i % num_gpus]
            print(f"\nExperiment {i+1}/{len(EXPERIMENTS)} (GPU {gpu_id}):")
            print(f"  Label: {exp['label']}")
            print(f"  Smoothing: {exp['use_smoothing']}, {exp['smoothing_method']}, alpha={exp['smoothing_alpha']}")
            print(f"  Cache: interval={exp.get('cache_interval', 5)}, max_order={exp.get('max_order', 2)}, first_enhance={exp.get('first_enhance', 3)}")
            print(f"  Output: {BASE_OUTDIR}/{exp['outdir_suffix']}")
        return

    # Run experiments in parallel batches
    completed = 0
    failed = 0

    for batch_start in range(0, len(EXPERIMENTS), max_parallel):
        batch = EXPERIMENTS[batch_start:batch_start + max_parallel]
        processes = []
        result_queue = Queue()

        print(f"\n\n{'#'*60}")
        print(f"Starting batch {batch_start//max_parallel + 1}/{(len(EXPERIMENTS) + max_parallel - 1)//max_parallel}")
        print(f"Experiments in this batch: {len(batch)}")
        print(f"{'#'*60}\n")

        # Create and start all processes for this batch
        for j, exp in enumerate(batch):
            gpu_id = gpu_list[j % num_gpus]
            p = Process(
                target=run_single_experiment,
                args=(exp, gpu_id, args.project_root, result_queue)
            )
            p.start()
            processes.append(p)
            print(f"[INFO] Started process for '{exp['label']}' on GPU {gpu_id} (PID: {p.pid})")

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Collect results
        while not result_queue.empty():
            status, label, gpu = result_queue.get()
            if status == 'success':
                completed += 1
            else:
                failed += 1

        print(f"\n[INFO] Batch completed. Success: {completed}, Failed: {failed}")

    print(f"\n\n{'='*60}")
    print(f"All experiments completed!")
    print(f"Success: {completed}, Failed: {failed}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
