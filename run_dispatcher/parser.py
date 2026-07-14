"""Argument parser for the unified dispatcher."""

from __future__ import annotations

import argparse

from .model_config import TaskType

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Unified inference dispatcher for Cache4Diffusion.",
        epilog="Run `python run.py --list` to see all available models grouped by task.",
    )
    p.add_argument("--list", action="store_true",
                   help="List all registered models (grouped by task) and exit.")
    p.add_argument("--model", type=str, help="Model name (see --list).")
    p.add_argument("--task", type=str, choices=[t.value for t in TaskType],
                   help="Filter --list by task, or assert the task for --model.")

    # What to run
    g = p.add_mutually_exclusive_group()
    g.add_argument("--prompt", type=str, help="Single text prompt.")
    g.add_argument("--prompt_file", type=str, help="File with one prompt per line (batch).")
    p.add_argument("--mode", choices=["single", "batch"], default=None,
                   help="single (one --prompt) or batch (--prompt_file). "
                        "Default: batch if --prompt_file given, else single.")
    p.add_argument("--input_image", type=str, help="Input image for image-editing models.")
    p.add_argument("--mask_path", type=str, help="Mask image for inpainting (flux-dev-fill).")
    p.add_argument("--dataset_path", type=str, help="GEdit-Bench dataset path (qwen_edit).")
    p.add_argument("--negative_prompt", type=str, help="Negative prompt (forwarded where supported).")

    # Generation params (common vocabulary)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dtype", type=str, default=None,
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--guidance", type=float, default=None,
                   help="Guidance scale (mapped to each model's own arg, e.g. "
                        "guidance_scale / true_cfg_scale / embedded-cfg-scale / guidance).")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--outdir", type=str, default=None, help="Output directory.")
    p.add_argument("--prefix", type=str, default=None, help="Output filename prefix.")
    p.add_argument("--num_images", type=int, default=None,
                   help="Number of outputs (maps to max_images / num_images_per_prompt / max_videos).")

    # Video-only
    p.add_argument("--video_length", type=int, default=None, help="Number of frames (video).")
    p.add_argument("--video_size", type=int, nargs=2, default=None, metavar=("HEIGHT", "WIDTH"),
                   help="Video resolution: HEIGHT WIDTH (video).")
    p.add_argument("--fps", type=int, default=None, help="Output video fps.")

    # Caching knobs (common vocabulary)
    p.add_argument("--cache_interval", type=int, default=None,
                   help="Cache fresh-threshold / interval.")
    p.add_argument("--cache_max_order", type=int, default=None, help="Taylor max order.")
    p.add_argument("--cache_first_enhance", type=int, default=None, help="First-enhance step.")
    p.add_argument("--no_cache", action="store_true",
                   help="Disable TaylorSeer caching (baseline run).")
    p.add_argument("--use_smoothing", action="store_true", help="Enable cached-value smoothing.")
    p.add_argument("--smoothing_alpha", type=float, default=None)
    p.add_argument("--smoothing_method", type=str, default=None,
                   choices=["exponential", "moving_average"])
    p.add_argument("--english_only", action="store_true",
                   help="GEdit-Bench: process English tasks only (qwen_edit).")

    # Paths / environment
    p.add_argument("--model_path", type=str, default=None,
                   help="Override the model checkpoint path/dir.")
    p.add_argument("--flux_t5_root", type=str, default=None, help="T5 root for raw FLUX repo.")
    p.add_argument("--flux_clip_root", type=str, default=None, help="CLIP root for raw FLUX repo.")
    p.add_argument("--conda_env", type=str, default=None,
                   help="Override the per-model conda environment.")
    p.add_argument("--conda_sh", type=str, default=None,
                   help="Path to conda.sh (default: auto / $CONDA_SH).")
    p.add_argument("--python", type=str, default=None,
                   help="Explicit interpreter (skips `conda activate`).")
    p.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES value.")
    p.add_argument("--nproc", type=int, default=None, help="torchrun --nproc_per_node.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the exact command + env without running it.")
    p.add_argument("--no_validate", action="store_true",
                   help="Do not check that entry scripts / model paths exist on disk.")

    # ── Unified output layout + post-generation evaluation ─────────────────
    # When --outdir is NOT given, images go to:
    #   {outdir_root}/{model}/[{variant}/]{method}/{config}/
    # where config encodes the acceleration knobs, e.g. S50_N5O1F3A0.
    # An explicit --outdir overrides the whole layout (used literally).
    p.add_argument("--outdir_root", type=str, default="outputs",
                   help="Root dir for the unified output layout (default: outputs).")
    p.add_argument("--variant", type=str, default=None,
                   help="Optional variant segment in the path (e.g. lora-animation2k_v1, quant-nf4).")
    p.add_argument("--method", type=str, default=None,
                   help="Method token for the path (default: baseline | taylorseer). "
                        "Override only for labelling; run.py only runs TaylorSeer today.")

    p.add_argument("--eval", action="store_true",
                   help="After successful generation, evaluate images in the `eval` conda env "
                        "(CLIP/ImageReward always; PSNR/SSIM/LPIPS when a reference is available).")
    p.add_argument("--eval_reference_folder", "--eval-ref", dest="eval_reference_folder",
                   type=str, default=None,
                   help="Reference folder for PSNR/SSIM/LPIPS. If omitted, auto-use the sibling "
                        "baseline/{Ssteps} folder when it exists; otherwise reference-free.")
    p.add_argument("--eval_prompt_file", type=str, default=None,
                   help="Prompt file for CLIP/ImageReward. Default: the generation prompt file "
                        "(or the single --prompt); fallback assets/prompts/DrawBench200.txt.")
    p.add_argument("--eval_gpu", type=str, default=None,
                   help="CUDA_VISIBLE_DEVICES for the eval step (default: same as --gpu).")
    p.add_argument("--eval_env", type=str, default="eval",
                   help="Conda environment for evaluation (default: eval).")
    p.add_argument("--eval_out", type=str, default=None,
                   help="Where to write the metrics file. Default: <outdir>/evaluation_results.txt.")

    # ── GEdit-Bench evaluation (editing models; evaluate_gedit.py / VIEScore) ──
    # Only consulted when the model's eval_kind == "gedit" (flux_kontext_gedit,
    # qwen_edit). VIEScore pulls the original images + edit instructions from the
    # GEdit-Bench dataset (GEDIT_DATASET_PATH / --dataset_path), so there is no
    # prompt file / reference folder here.
    p.add_argument("--gedit_backbone", type=str, default="qwen25vl", choices=["gpt4o", "qwen25vl"],
                   help="VIEScore backbone for evaluate_gedit.py (default qwen25vl, local).")
    p.add_argument("--gedit_language", type=str, default="en", choices=["all", "en", "cn"],
                   help="GEdit instruction language to score (default en).")
    p.add_argument("--gedit_task_type", type=str, default="all",
                   help="GEdit task type (default all; e.g. color_alter, subject-add, ...).")
    p.add_argument("--qwen25vl_model_path", type=str, default=None,
                   help="Qwen2.5-VL-72B-Instruct-AWQ checkpoint for the qwen25vl backbone "
                        "(default: built-in DEFAULT_QWEN25VL_MODEL).")

    # ── Speed benchmark (latency + FLOPs) ─────────────────────────────────
    # Runs a SINGLE generation and measures real wall-clock latency (after
    # warmup) plus total transformer FLOPs (via calflops, in a separate
    # profiling pass). Report → <outdir>/benchmark.txt. Forces single mode.
    p.add_argument("--benchmark", action="store_true",
                   help="Benchmark a single generation: measure latency + transformer FLOPs "
                        "and write <outdir>/benchmark.txt. Incompatible with --eval.")
    p.add_argument("--benchmark_warmup", type=int, default=1,
                   help="Number of untimed warmup generations before timing (default 1).")
    p.add_argument("--benchmark_runs", type=int, default=1,
                   help="Number of timed latency runs; report their mean (default 1).")
    p.add_argument("--benchmark_report", type=str, default=None,
                   help="Where to write the benchmark report. Default: <outdir>/benchmark.txt.")
    return p


# --------------------------------------------------------------------------- #

