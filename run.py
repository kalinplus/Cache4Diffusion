#!/usr/bin/env python3
"""
Unified inference dispatcher for Cache4Diffusion.

One Python entry point to run *any* of the per-model pipelines in this repo:
  - image generation   (flux_diffusers, flux, hunyuan_image, qwen_image, qwen)
  - image editing      (flux_kontext [single-image], flux_kontext_gedit / qwen_edit [GEdit-Bench])
  - video generation   (hunyuan_video)

It does NOT re-implement any model logic.  Each model already has its own
working entry-point script under its directory; this dispatcher simply:

  1. maps a *common* vocabulary (steps / guidance / size / cache knobs ...) onto
     each entry script's own argument names (they differ wildly between models),
  2. picks the right conda env, working dir, PYTHONPATH and launcher
     (plain ``python`` vs ``torchrun``),
  3. builds the exact command and runs it via ``conda activate`` + subprocess.

Because the dispatcher only uses the Python standard library, ``--list`` and
``--dry-run`` work on any interpreter (no torch / diffusers import), which is
what lets us validate the wiring without GPUs or model weights.

Quick examples:

    python run.py --list
    python run.py --list --task image_gen
    python run.py --model flux_diffusers --dry-run --prompt "a cat" --steps 50
    python run.py --model qwen_image --gpu 0 --prompt "a cat" --steps 50
    python run.py --model hunyuan_video --task video_gen \\
        --prompt "a cat walks on grass" --video_length 65 --video_size 544 960
    # image editing over GEdit-Bench (generation + VIEScore eval):
    python run.py --model flux_kontext_gedit --dataset_path /path/to/GEdit-Bench --eval
    python run.py --model qwen_edit --dataset_path /path/to/GEdit-Bench --eval
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

# Project root (the directory that contains run.py / the 6 model dirs).
ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #
class TaskType(str, enum.Enum):
    IMAGE_GEN = "image_gen"
    IMAGE_EDIT = "image_edit"
    VIDEO_GEN = "video_gen"

    @property
    def label(self) -> str:
        return {
            TaskType.IMAGE_GEN: "图片生成 (image generation)",
            TaskType.IMAGE_EDIT: "图片编辑 (image editing)",
            TaskType.VIDEO_GEN: "视频生成 (video generation)",
        }[self]


@dataclasses.dataclass
class ModelRunner:
    """Declarative description of how to launch one model pipeline.

    ``arg_map`` / ``env_map`` / ``flags`` map a *common* parameter key onto the
    target script's own CLI argument / env var / store_true flag.  Only keys
    present in these maps are forwarded, so we never pass an argument a script
    does not understand.
    """

    name: str
    task: TaskType
    description: str

    # Conda environment to activate before running.
    conda_env: str

    # Entry scripts, given RELATIVE to ``workdir``.  ``entry_single`` accepts a
    # single --prompt; ``entry_batch`` accepts --prompt_file.  They may be the
    # same script (some batch scripts also accept a single --prompt).
    entry_single: Optional[str] = None
    entry_batch: Optional[str] = None

    # Working directory (RELATIVE to ROOT) to cd into before launching, and the
    # directories (RELATIVE to ROOT) to place on PYTHONPATH.
    workdir: str = "."
    pythonpath: Sequence[str] = ()

    # "python" -> plain interpreter; "torchrun" -> torchrun --standalone.
    launcher: str = "python"
    nproc: int = 1

    # common-key -> CLI arg name (without leading --).  A list/tuple value is
    # spread as ``--arg v1 v2`` (nargs).  True -> store_true, False -> skipped.
    arg_map: Dict[str, str] = dataclasses.field(default_factory=dict)
    # common-key -> environment variable name (string value).
    env_map: Dict[str, str] = dataclasses.field(default_factory=dict)
    # common-key -> environment variable name, where the value is a bool that
    # gets serialized as "True"/"False" (e.g. flux_diffusers's USE_SMOOTHING).
    env_bool_map: Dict[str, str] = dataclasses.field(default_factory=dict)
    # common-key -> store_true flag name (without --).  Forwarded only when the
    # common value is truthy.
    flags: Dict[str, str] = dataclasses.field(default_factory=dict)
    # Always-on argument tokens, e.g. ["--english_only"].
    extra_args: Sequence[str] = ()
    # Static env vars always set for this model.
    env_static: Dict[str, str] = dataclasses.field(default_factory=dict)

    # Store_true flag that turns caching ON (e.g. "--use_taylor").  Auto-added
    # unless the user requests --no_cache.
    cache_flag: Optional[str] = None

    # How the model checkpoint reaches the script:
    #   model_path_arg  -> "--<arg> <path>"
    #   model_path_env  -> environment variable = <path>
    model_path_arg: Optional[str] = None
    model_path_env: Optional[str] = None

    # Fixed --model_name value + the arg name that carries it (if any).
    model_name: Optional[str] = None
    model_name_arg: Optional[str] = None

    # Per-model sensible defaults for common keys.
    defaults: Dict[str, Any] = dataclasses.field(default_factory=dict)

    output_ext: str = "png"  # png | jpg | mp4
    dtype_default: str = "bfloat16"

    # True once the entry_single script accepts --benchmark / --benchmark_warmup
    # / --benchmark_runs / --benchmark_report and runs the shared
    # cache4diffusion_bench harness (latency + FLOPs) for a single generation.
    supports_benchmark: bool = False

    # Optional(env_builder) -> dict: extra env vars computed from params
    # (used by the raw FLUX repo, which loads weights via FLUX_MODEL / FLUX_AE /
    # T5_MODEL_PATH / CLIP_MODEL_PATH env vars).
    env_builder: Optional[Callable[[Dict[str, Any], Dict[str, str]], Dict[str, str]]] = None

    # Post-generation evaluation routing.
    #   eval_kind  : "t2i" -> evaluate.py (CLIP/ImageReward/PSNR/SSIM/LPIPS);
    #                "gedit" -> per-directory evaluate_gedit.py (VIEScore).
    #   eval_entry : gedit eval script, RELATIVE to ROOT (None -> evaluate/evaluate.py).
    #   eval_cwd   : dir RELATIVE to ROOT to run the gedit eval from, so the
    #                script's `from viescore import VIEScore` resolves (default: workdir).
    eval_kind: str = "t2i"
    eval_entry: Optional[str] = None
    eval_cwd: Optional[str] = None

    note: str = ""


# --------------------------------------------------------------------------- #
# Default model checkpoint paths (overridable via --model_path / env).
# These follow the paths used by the per-model shell scripts; adjust to your
# local layout with --model_path or the MODEL_PATH / MODEL_PATH_<NAME> env var.
# --------------------------------------------------------------------------- #
DEFAULT_MODEL_PATHS: Dict[str, str] = {
    "flux_diffusers": "/mnt/workspace/hkl/models/black-forest-labs/FLUX.1-dev",
    "flux": "/mnt/workspace/hkl/models/black-forest-labs/FLUX.1-dev",
    "flux_kontext": "/mnt/workspace/hkl/models/black-forest-labs/FLUX.1-Kontext-dev",
    "flux_kontext_gedit": "/mnt/workspace/hkl/models/black-forest-labs/FLUX.1-Kontext-dev",
    "hunyuan_image": "/mnt/workspace/hkl/models/tencent/HunyuanImage-2.1",
    "qwen_image": "/mnt/workspace/hkl/models/Qwen/Qwen-Image",
    "qwen": "/mnt/workspace/hkl/models/Qwen/Qwen-Image",
    "qwen_edit": "/mnt/workspace/hkl/models/Qwen/Qwen-Image-Edit",
    "hunyuan_video": "/mnt/workspace/hkl/models/hunyuanvideo-community/HunyuanVideo",
}

# Default T5 / CLIP roots for the raw FLUX repo (flux / flux_kontext).
DEFAULT_FLUX_T5_ROOT = "/mnt/workspace/hkl/models/google/t5-v1_1-xxl"
DEFAULT_FLUX_CLIP_ROOT = "/mnt/workspace/hkl/models/openai/clip-vit-large-patch14"
DEFAULT_GEDIT_DATASET = "/mnt/data0/datasets/stepfun-ai/GEdit-Bench"
# VIEScore backbone used by evaluate_gedit.py (gedit eval). Local AWQ checkpoint.
DEFAULT_QWEN25VL_MODEL = "/mnt/data0/Qwen/Qwen2.5-VL-72B-Instruct-AWQ"

# Common negative prompt used by several models when none is supplied.
DEFAULT_NEGATIVE_PROMPT = "blurry, low resolution, bad anatomy, watermark"

# Common keys that configure caching; suppressed as a group under --no_cache.
CACHE_KEYS = {"cache_interval", "cache_max_order", "cache_first_enhance"}


# --------------------------------------------------------------------------- #
# Helpers for the raw FLUX repo (weights are addressed via env vars).
# --------------------------------------------------------------------------- #
def _flux_raw_env_builder(flow_filename: str) -> Callable[[Dict[str, Any], Dict[str, str]], Dict[str, str]]:
    """Build FLUX_MODEL / FLUX_AE / T5_MODEL_PATH / CLIP_MODEL_PATH env vars."""

    def _build(params: Dict[str, Any], env: Dict[str, str]) -> Dict[str, str]:
        root = params.get("model_path") or ""
        t5 = params.get("flux_t5_root") or DEFAULT_FLUX_T5_ROOT
        clip = params.get("flux_clip_root") or DEFAULT_FLUX_CLIP_ROOT
        if root:
            env["FLUX_MODEL"] = str(Path(root) / flow_filename)
            env["FLUX_AE"] = str(Path(root) / "ae.safetensors")
        env["T5_MODEL_PATH"] = str(t5)
        env["CLIP_MODEL_PATH"] = str(clip)
        return env

    return _build


# --------------------------------------------------------------------------- #
# The registry — the single source of truth for "how to run each model".
# --------------------------------------------------------------------------- #
REGISTRY: Dict[str, ModelRunner] = {

    # ── image generation ────────────────────────────────────────────────────
    "flux_diffusers": ModelRunner(
        name="flux_diffusers",
        task=TaskType.IMAGE_GEN,
        description="FLUX.1-dev text-to-image (diffusers + TaylorSeer) — flux_diffusers/",
        supports_benchmark=True,
        conda_env="infer",
        entry_single="taylorseer_flux/diffusers_taylorseer_flux.py",
        entry_batch="taylorseer_flux/batch_infer.py",
        workdir="flux_diffusers",
        pythonpath=("flux_diffusers", "."),
        arg_map={
            "prompt": "prompt", "prompt_file": "prompt_file",
            "steps": "steps", "seed": "seed", "dtype": "dtype",
            "guidance": "guidance_scale",
            "width": "width", "height": "height",
            "outdir": "outdir", "prefix": "prefix",
            "num_images": "max_images",
        },
        env_map={
            "cache_interval": "FRESH_THRESHOLD",
            "cache_max_order": "MAX_ORDER",
            "cache_first_enhance": "FIRST_ENHANCE",
            "smoothing_alpha": "SMOOTHING_ALPHA",
            "smoothing_method": "SMOOTHING_METHOD",
        },
        env_bool_map={"use_smoothing": "USE_SMOOTHING"},
        model_path_arg="model",
        defaults={"guidance": 7.5, "dtype": "float16", "steps": 50, "seed": 42,
                  "width": 1024, "height": 1024, "cache_interval": 5,
                  "cache_max_order": 1, "cache_first_enhance": 1},
        output_ext="png",
        dtype_default="float16",
        note="Caching knobs (FRESH_THRESHOLD/MAX_ORDER/...) are read from env vars.",
    ),

    "flux": ModelRunner(
        name="flux",
        task=TaskType.IMAGE_GEN,
        description="FLUX.1-dev text-to-image (raw BFL repo + TaylorSeer) — flux/taylorseer/",
        supports_benchmark=True,
        conda_env="infer",
        entry_single="sample.py",
        entry_batch="sample.py",  # sample.py reads --prompt_file; single prompt -> temp file
        workdir="flux/taylorseer/src",
        pythonpath=(),
        launcher="python",
        arg_map={
            "prompt_file": "prompt_file",
            "steps": "num_steps", "seed": "seed", "guidance": "guidance",
            "width": "width", "height": "height",
            "outdir": "output_dir", "num_images": "num_images_per_prompt",
            "cache_interval": "interval", "cache_max_order": "max_order",
            "cache_first_enhance": "first_enhance",
        },
        model_name="flux-dev",
        model_name_arg="model_name",
        model_path_env="__flux_root__",  # placeholder; real vars come from env_builder
        env_builder=_flux_raw_env_builder("flux1-dev.safetensors"),
        defaults={"guidance": 3.5, "steps": 50, "seed": 42,
                  "width": 1024, "height": 1024, "cache_interval": 4,
                  "cache_max_order": 0, "cache_first_enhance": 1},
        output_ext="jpg",
        dtype_default="bfloat16",
        note="Weights via env FLUX_MODEL/FLUX_AE/T5_MODEL_PATH/CLIP_MODEL_PATH; "
             "single --prompt is written to a temp prompt file.",
    ),

    "hunyuan_image": ModelRunner(
        name="hunyuan_image",
        task=TaskType.IMAGE_GEN,
        description="HunyuanImage-2.1 high-res text-to-image (TaylorSeer Lite) — HunyuanImage-2.1/",
        supports_benchmark=True,
        conda_env="infer",
        # The batch script accepts --prompt OR --prompt_file, so reuse it for both.
        entry_single="run_hyimage_taylorseer_lite_batch.py",
        entry_batch="run_hyimage_taylorseer_lite_batch.py",
        workdir="HunyuanImage-2.1",
        pythonpath=("HunyuanImage-2.1", "."),
        arg_map={
            "prompt": "prompt", "prompt_file": "prompt_file",
            "steps": "steps", "seed": "seed",
            "guidance": "guidance_scale",
            "width": "width", "height": "height",
            "outdir": "outdir", "prefix": "prefix",
            "smoothing_alpha": "smoothing_alpha", "smoothing_method": "smoothing_method",
        },
        flags={"use_smoothing": "use_smoothing"},
        cache_flag="--use_taylorseer_lite",
        model_path_env="HUNYUANIMAGE_V2_1_MODEL_ROOT",
        model_name="hunyuanimage-v2.1",
        model_name_arg="model_name",
        defaults={"guidance": 3.5, "steps": 50, "seed": 649151,
                  "width": 2048, "height": 2048, "smoothing_alpha": 0.8,
                  "smoothing_method": "exponential", "shift": 5.0},
        output_ext="png",
        dtype_default="bfloat16",
        note="Model root via env HUNYUANIMAGE_V2_1_MODEL_ROOT. dtype is bf16 (fixed).",
    ),

    "qwen_image": ModelRunner(
        name="qwen_image",
        task=TaskType.IMAGE_GEN,
        description="Qwen-Image text-to-image (diffusers + TaylorSeer) — qwen_image/",
        supports_benchmark=True,
        conda_env="infer",
        entry_single="diffusers_taylorseer_qwen_image.py",
        entry_batch="batch_infer.py",
        workdir="qwen_image/taylorseer_qwen_image",
        pythonpath=(".", "..", "../..", "flux_diffusers"),  # flux_diffusers holds the shared taylorseer_core package
        arg_map={
            "prompt": "prompt", "prompt_file": "prompt_file",
            "steps": "steps", "seed": "seed", "dtype": "dtype",
            "guidance": "true_cfg_scale",
            "width": "width", "height": "height",
            "outdir": "outdir", "prefix": "prefix",
            "num_images": "max_images",
        },
        env_map={"smoothing_alpha": "SMOOTHING_ALPHA",
                 "smoothing_method": "SMOOTHING_METHOD"},
        env_bool_map={"use_smoothing": "USE_SMOOTHING"},
        cache_flag="--use_taylor",
        model_path_arg="model",
        defaults={"guidance": 7.5, "dtype": "bfloat16", "steps": 50, "seed": 42,
                  "width": 1328, "height": 1328, "smoothing_alpha": 0.85,
                  "smoothing_method": "exponential"},
        output_ext="png",
        dtype_default="bfloat16",
        note="Pipeline loads with device_map='cuda'; uses --use_taylor + true_cfg_scale.",
    ),

    "qwen": ModelRunner(
        name="qwen",
        task=TaskType.IMAGE_GEN,
        description="Qwen-Image text-to-image (raw repo + TaylorSeer) — qwen/taylorseer/",
        supports_benchmark=True,
        conda_env="infer",
        entry_single="sample.py",
        entry_batch="sample.py",  # reads --prompt_file; single prompt -> temp file
        workdir="qwen/taylorseer",
        pythonpath=(".",),
        arg_map={
            "prompt_file": "prompt_file", "input_image": "input_image",
            "steps": "num_steps", "seed": "seed", "guidance": "guidance_scale",
            "width": "width", "height": "height",
            "outdir": "output_dir", "num_images": "num_images_per_prompt",
            "cache_interval": "interval", "cache_max_order": "max_order",
            "cache_first_enhance": "first_enhance",
        },
        model_name="qwen-image",
        model_name_arg="model_name",
        model_path_env="QWEN_IMAGE_MODEL_PATH",
        defaults={"guidance": 1.0, "steps": 50, "seed": 0,
                  "width": 1328, "height": 1328, "cache_interval": 6,
                  "cache_max_order": 2, "cache_first_enhance": 3},
        output_ext="jpg",
        dtype_default="bfloat16",
        note="Model path via env QWEN_IMAGE_MODEL_PATH. single --prompt -> temp prompt file.",
    ),

    # ── image editing ───────────────────────────────────────────────────────
    "flux_kontext": ModelRunner(
        name="flux_kontext",
        task=TaskType.IMAGE_EDIT,
        description="FLUX.1-Kontext-dev image editing (raw BFL repo + TaylorSeer) — flux/taylorseer/",
        supports_benchmark=True,
        conda_env="infer",
        entry_single="sample.py",
        entry_batch="sample.py",
        workdir="flux/taylorseer/src",
        pythonpath=(),
        arg_map={
            "prompt_file": "prompt_file", "input_image": "input_image",
            "mask_path": "mask_path",
            "steps": "num_steps", "seed": "seed", "guidance": "guidance",
            "width": "width", "height": "height",
            "outdir": "output_dir", "num_images": "num_images_per_prompt",
            "cache_interval": "interval", "cache_max_order": "max_order",
            "cache_first_enhance": "first_enhance",
        },
        model_name="flux-dev-kontext",
        model_name_arg="model_name",
        model_path_env="__flux_root__",
        env_builder=_flux_raw_env_builder("flux1-kontext-dev.safetensors"),
        defaults={"guidance": 3.5, "steps": 50, "seed": 42,
                  "width": 1360, "height": 768, "cache_interval": 4,
                  "cache_max_order": 0, "cache_first_enhance": 1},
        output_ext="jpg",
        dtype_default="bfloat16",
        note="Single-image edit (--input_image; --mask_path for inpaint). "
             "For the GEdit-Bench sweep use flux_kontext_gedit.",
    ),

    "flux_kontext_gedit": ModelRunner(
        name="flux_kontext_gedit",
        task=TaskType.IMAGE_EDIT,
        description="FLUX.1-Kontext-dev image editing over GEdit-Bench (raw BFL repo + TaylorSeer, torchrun) — flux/taylorseer/",
        supports_benchmark=True,
        conda_env="infer",
        entry_single="sample_gedit.py",
        entry_batch="sample_gedit.py",
        workdir="flux/taylorseer/src",
        pythonpath=(),
        launcher="torchrun",
        nproc=1,
        arg_map={
            "dataset_path": "dataset_path",
            "steps": "num_steps", "seed": "seed", "guidance": "guidance",
            "outdir": "output_dir",
            "cache_interval": "interval", "cache_max_order": "max_order",
            "cache_first_enhance": "first_enhance",
        },
        env_map={"smoothing_alpha": "SMOOTHING_ALPHA",
                 "smoothing_method": "SMOOTHING_METHOD"},
        env_bool_map={"use_smoothing": "USE_SMOOTHING"},
        flags={"english_only": "english_only"},
        model_name="flux-dev-kontext",
        model_name_arg="model_name",
        model_path_env="__flux_root__",
        env_builder=_flux_raw_env_builder("flux1-kontext-dev.safetensors"),
        eval_kind="gedit",
        eval_entry="flux/taylorseer/evaluate_gedit.py",
        eval_cwd="flux/taylorseer",
        defaults={"guidance": 3.5, "steps": 50, "seed": 0,
                  "cache_interval": 9, "cache_max_order": 1, "cache_first_enhance": 3,
                  "dataset_path": DEFAULT_GEDIT_DATASET, "english_only": True,
                  "smoothing_alpha": 0.8, "smoothing_method": "exponential"},
        output_ext="png",
        dtype_default="bfloat16",
        note="GEdit-Bench editing sweep (torchrun). Weights via env "
             "FLUX_MODEL/FLUX_AE/T5_MODEL_PATH/CLIP_MODEL_PATH; "
             "--eval runs evaluate_gedit.py (VIEScore, default qwen25vl).",
    ),

    "qwen_edit": ModelRunner(
        name="qwen_edit",
        task=TaskType.IMAGE_EDIT,
        description="Qwen-Image-Edit instruction editing over GEdit-Bench (raw repo, torchrun) — qwen/taylorseer/",
        supports_benchmark=True,
        conda_env="infer",
        entry_single="sample_gedit.py",
        entry_batch="sample_gedit.py",
        workdir="qwen/taylorseer",
        pythonpath=(".",),
        launcher="torchrun",
        nproc=1,
        arg_map={
            "dataset_path": "dataset_path", "steps": "num_steps",
            "seed": "seed", "guidance": "guidance_scale",
            "outdir": "output_dir",
            "cache_interval": "interval", "cache_max_order": "max_order",
            "cache_first_enhance": "first_enhance",
        },
        env_map={"smoothing_alpha": "SMOOTHING_ALPHA",
                 "smoothing_method": "SMOOTHING_METHOD"},
        env_bool_map={"use_smoothing": "USE_SMOOTHING"},
        flags={"english_only": "english_only"},
        extra_args=(),
        # GEdit pipeline is selected by passing the checkpoint path as --model_name.
        model_name_arg="model_name",
        model_path_arg="model_name",
        eval_kind="gedit",
        eval_entry="qwen/taylorseer/evaluate_gedit.py",
        eval_cwd="qwen/taylorseer",
        defaults={"guidance": 1.0, "steps": 50, "seed": 0,
                  "cache_interval": 10, "cache_max_order": 2, "cache_first_enhance": 3,
                  "dataset_path": DEFAULT_GEDIT_DATASET, "english_only": True,
                  "smoothing_alpha": 0.8, "smoothing_method": "exponential"},
        output_ext="png",
        dtype_default="bfloat16",
        note="GEdit-Bench editing sweep (torchrun), run from inside qwen/taylorseer. "
             "Input images come from the dataset (--dataset_path), not a CLI file. "
             "--eval runs evaluate_gedit.py (VIEScore, default qwen25vl).",
    ),

    # ── video generation ────────────────────────────────────────────────────
    "hunyuan_video": ModelRunner(
        name="hunyuan_video",
        task=TaskType.VIDEO_GEN,
        description="HunyuanVideo text-to-video (diffusers + TaylorSeer) — hunyuan_video/taylorseer_hunyuan_video/",
        supports_benchmark=True,
        conda_env="hyv15",
        entry_single="diffusers_taylorseer_hunyuan_video.py",
        entry_batch="batch_infer.py",
        workdir="hunyuan_video/taylorseer_hunyuan_video",
        pythonpath=(".", "..", "../.."),
        arg_map={
            "prompt": "prompt", "prompt_file": "prompt_file",
            "steps": "infer-steps", "seed": "seed", "dtype": "dtype",
            "guidance": "embedded-cfg-scale",
            "video_size": "video-size", "video_length": "video-length",
            "fps": "fps",
            "outdir": "save-path", "prefix": "prefix",
            "num_images": "max_videos",
        },
        flags={"use_cpu_offload": "use_cpu_offload"},
        cache_flag="--use_taylor",
        model_path_arg="model",
        env_static={"TOKENIZERS_PARALLELISM": "false",
                    "DIFFUSERS_ATTN_BACKEND": "flash"},
        defaults={"guidance": 6.0, "dtype": "bfloat16", "steps": 50, "seed": 42,
                  "video_size": [720, 1280], "video_length": 129, "fps": 24},
        output_ext="mp4",
        dtype_default="bfloat16",
        note="--video_size is 'HEIGHT WIDTH'. Use bf16 (fp16 can yield NaN/black video).",
    ),
}


# --------------------------------------------------------------------------- #
# Argument parsing for the dispatcher itself
# --------------------------------------------------------------------------- #
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
# Resolution helpers
# --------------------------------------------------------------------------- #
def _abs(path: str) -> str:
    """Make ``path`` absolute relative to ROOT (unless already absolute)."""
    return str(Path(path).expanduser() if Path(path).is_absolute()
               else (ROOT / path).resolve())


def resolve_conda_sh(arg: Optional[str]) -> str:
    if arg:
        return arg
    env = os.environ.get("CONDA_SH")
    if env:
        return env
    # Common default for this machine.
    candidates = [
        "/mnt/workspace/hkl/miniconda3/etc/profile.d/conda.sh",
        os.path.expanduser("~/miniconda3/etc/profile.d/conda.sh"),
        os.path.expanduser("~/anaconda3/etc/profile.d/conda.sh"),
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return candidates[0]


def resolve_model_path(runner: ModelRunner, override: Optional[str]) -> str:
    if override:
        return override
    env_generic = os.environ.get("MODEL_PATH")
    if env_generic:
        return env_generic
    env_specific = os.environ.get(f"MODEL_PATH_{runner.name.upper()}")
    if env_specific:
        return env_specific
    return DEFAULT_MODEL_PATHS.get(runner.name, "")


def resolve_params(args: argparse.Namespace, runner: ModelRunner) -> Dict[str, Any]:
    """Merge CLI args over the runner's defaults into a flat common-param dict."""
    params: Dict[str, Any] = dict(runner.defaults)

    cli_map = {
        "prompt": args.prompt, "prompt_file": args.prompt_file,
        "input_image": args.input_image, "mask_path": args.mask_path,
        "dataset_path": args.dataset_path,
        "negative_prompt": args.negative_prompt,
        "steps": args.steps, "seed": args.seed, "dtype": args.dtype,
        "guidance": args.guidance,
        "width": args.width, "height": args.height,
        "outdir": args.outdir, "prefix": args.prefix,
        "num_images": args.num_images,
        "video_length": args.video_length,
        "video_size": args.video_size, "fps": args.fps,
        "cache_interval": args.cache_interval,
        "cache_max_order": args.cache_max_order,
        "cache_first_enhance": args.cache_first_enhance,
        "use_smoothing": args.use_smoothing or None,
        "smoothing_alpha": args.smoothing_alpha,
        "smoothing_method": args.smoothing_method,
        "english_only": args.english_only or None,
        "flux_t5_root": args.flux_t5_root, "flux_clip_root": args.flux_clip_root,
    }
    for k, v in cli_map.items():
        if v is not None:
            params[k] = v

    # model_path
    params["model_path"] = resolve_model_path(runner, args.model_path)

    # caching on/off
    params["use_cache"] = not args.no_cache

    # Speed benchmark: latency + FLOPs on a single generation.
    if getattr(args, "benchmark", False):
        params["benchmark"] = True
        params["benchmark_warmup"] = args.benchmark_warmup
        params["benchmark_runs"] = args.benchmark_runs
        # benchmark_report is resolved in main() once outdir is known.
    return params


def resolve_entry(runner: ModelRunner, mode: str) -> str:
    if mode == "batch":
        entry = runner.entry_batch or runner.entry_single
    else:
        entry = runner.entry_single or runner.entry_batch
    if not entry:
        raise SystemExit(f"[run.py] model '{runner.name}' has no entry script for mode '{mode}'.")
    return entry


def resolve_mode(args: argparse.Namespace) -> str:
    if args.mode:
        return args.mode
    return "batch" if args.prompt_file else "single"


# --------------------------------------------------------------------------- #
# Unified output layout:  {outdir_root}/{model}/[{variant}/]{method}/{config}/
# --------------------------------------------------------------------------- #
def _format_alpha(alpha: Any) -> str:
    """Render a smoothing alpha for the path: 0 / 0.8 / 0.85 ..."""
    try:
        a = float(alpha) if alpha is not None else 0.0
    except (TypeError, ValueError):
        return "0"
    return "0" if a == 0 else f"{a:g}"


def _runner_forwards_cache_knobs(runner: ModelRunner) -> bool:
    """True if this model's arg_map/env_map actually carries N/O/F cache knobs."""
    mapped = set(runner.arg_map) | set(runner.env_map)
    return any(k in mapped for k in CACHE_KEYS)


def _config_str(runner: ModelRunner, params: Dict[str, Any], method: str) -> str:
    """Build the {config} segment: S{steps}[_N{}O{}F{}][A{alpha}]."""
    steps = params.get("steps")
    s = f"S{steps}" if steps is not None else "S0"
    if method == "baseline":
        return s
    use_cache = params.get("use_cache", True)
    # Append N/O/F only when caching is on AND the runner forwards those knobs.
    if use_cache and _runner_forwards_cache_knobs(runner) and params.get("cache_interval") is not None:
        n = params.get("cache_interval")
        o = params.get("cache_max_order")
        f = params.get("cache_first_enhance")
        s += f"_N{n}O{o}F{f}"
    if params.get("use_smoothing"):
        s += f"A{_format_alpha(params.get('smoothing_alpha'))}"
    return s


def _resolve_method(params: Dict[str, Any], args: argparse.Namespace) -> str:
    if args.method:
        return args.method
    return "baseline" if not params.get("use_cache", True) else "taylorseer"


def compute_layout(runner: ModelRunner, params: Dict[str, Any],
                   args: argparse.Namespace, method: Optional[str] = None) -> Dict[str, Any]:
    """Compute the unified-layout directory for a run (or a forced method).

    Returns root/model/variant/method/config and the absolute ``leaf`` path.
    Passing ``method='baseline'`` yields the sibling baseline leaf used as the
    auto reference for evaluation.
    """
    root = args.outdir_root if args.outdir_root is not None else "outputs"
    model = runner.name
    variant = args.variant
    if method is None:
        method = _resolve_method(params, args)
    config = _config_str(runner, params, method)
    parts = [root, model]
    if variant:
        parts.append(variant)
    parts.append(method)
    parts.append(config)
    leaf = _abs("/".join(parts))
    return {"root": root, "model": model, "variant": variant,
            "method": method, "config": config, "leaf": leaf}


def resolve_unified_outdir(runner: ModelRunner, params: Dict[str, Any],
                           args: argparse.Namespace) -> str:
    """Default --outdir under the unified layout (absolute)."""
    return compute_layout(runner, params, args)["leaf"]


def resolve_eval_reference(runner: ModelRunner, params: Dict[str, Any],
                           args: argparse.Namespace) -> tuple:
    """Return (reference_folder_or_None, source) for the eval step.

    source is one of 'explicit' / 'auto' / 'none'.
    """
    if args.eval_reference_folder:
        return _abs(args.eval_reference_folder), "explicit"
    baseline = compute_layout(runner, params, args, method="baseline")["leaf"]
    if Path(baseline).exists():
        return baseline, "auto"
    return None, "none"



# --------------------------------------------------------------------------- #
# Command construction
# --------------------------------------------------------------------------- #
def _maybe_temp_prompt_file(runner: ModelRunner, params: Dict[str, Any]) -> None:
    """If the chosen entry only accepts --prompt_file but the user gave --prompt,
    write the prompt to a temp file and expose it as prompt_file."""
    accepts_prompt = "prompt" in runner.arg_map
    accepts_file = "prompt_file" in runner.arg_map
    if accepts_prompt or not accepts_file:
        return
    if params.get("prompt_file"):
        return
    prompt = params.get("prompt")
    if not prompt:
        return
    tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    tmp.write(prompt + "\n")
    tmp.close()
    params["prompt_file"] = tmp.name
    params["__temp_prompt_file"] = tmp.name


def build_argv(runner: ModelRunner, params: Dict[str, Any]) -> List[str]:
    argv: List[str] = []

    # Model path (CLI arg form).
    if runner.model_path_arg and runner.model_path_env == "__flux_root__":
        # raw FLUX repo: path goes to env vars, not a CLI arg.
        pass
    elif runner.model_path_arg and params.get("model_path"):
        argv += [f"--{runner.model_path_arg}", str(params["model_path"])]

    # Fixed model_name (unless model_path already occupies the same arg, e.g. qwen_edit).
    if (runner.model_name and runner.model_name_arg
            and runner.model_name_arg != runner.model_path_arg):
        argv += [f"--{runner.model_name_arg}", runner.model_name]

    # store_true flags.
    for key, flag in runner.flags.items():
        if params.get(key):
            argv.append(f"--{flag}")

    # Caching on/off flag.
    if runner.cache_flag and params.get("use_cache", True):
        argv.append(runner.cache_flag)

    # Mapped args.
    for key, arg in runner.arg_map.items():
        if key in CACHE_KEYS and not params.get("use_cache", True):
            continue  # baseline run: do not inject any caching knobs
        val = params.get(key)
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            argv += [f"--{arg}"] + [str(v) for v in val]
        elif val is True:
            argv.append(f"--{arg}")
        elif val is False:
            continue
        else:
            argv += [f"--{arg}", str(val)]

    # Speed benchmark knobs (forwarded verbatim; entry scripts accept these
    # only when supports_benchmark is True, which main() gates on).
    if params.get("benchmark"):
        argv.append("--benchmark")
        argv += ["--benchmark_warmup", str(params.get("benchmark_warmup", 1))]
        argv += ["--benchmark_runs", str(params.get("benchmark_runs", 1))]
        report = params.get("benchmark_report")
        if report:
            argv += ["--benchmark_report", str(report)]

    # Always-on extras.
    argv += list(runner.extra_args)
    return argv


def build_env(runner: ModelRunner, params: Dict[str, Any],
              args: argparse.Namespace) -> Dict[str, str]:
    env: Dict[str, str] = {}

    # Mapped env vars.
    for key, var in runner.env_map.items():
        if key in CACHE_KEYS and not params.get("use_cache", True):
            continue  # baseline run: do not inject any caching knobs
        val = params.get(key)
        if val is None:
            continue
        if isinstance(val, bool):
            env[var] = "True" if val else "False"
        else:
            env[var] = str(val)

    # Boolean env vars (serialized as "True"/"False").
    for key, var in runner.env_bool_map.items():
        val = params.get(key)
        if val is None:
            continue
        env[var] = "True" if val else "False"

    # Model path (env var form).
    if runner.model_path_env and runner.model_path_env != "__flux_root__" and params.get("model_path"):
        env[runner.model_path_env] = str(params["model_path"])

    # Static env vars.
    env.update(runner.env_static)

    # Custom env builder (raw FLUX repo).
    if runner.env_builder:
        env = runner.env_builder(params, env)

    # PYTHONPATH.
    pp_parts = [_abs(d) for d in runner.pythonpath]
    # The shared benchmark helper (cache4diffusion_bench.py) lives at ROOT; make
    # sure ROOT is importable from any workdir when benchmarking.
    if params.get("benchmark"):
        root_abs = str(ROOT)
        if root_abs not in pp_parts:
            pp_parts.insert(0, root_abs)
    existing = os.environ.get("PYTHONPATH")
    if existing:
        pp_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pp_parts)

    # CUDA devices.
    gpu = args.gpu if args.gpu is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Helpful defaults inherited from the repo's .bashrc conventions.
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("DIFFUSERS_ATTN_BACKEND", "flash")
    env.setdefault("XDG_CACHE_HOME", "/data/public/.cache")
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    return env


def build_shell_command(runner: ModelRunner, entry: str, argv: List[str],
                        conda_env: str, args: argparse.Namespace) -> str:
    workdir_abs = _abs(runner.workdir)
    entry_abs = str((Path(workdir_abs) / entry).resolve())

    if args.python:
        # Explicit interpreter: skip conda activation entirely.
        launcher_prefix = f"{shlex.quote(args.python)} {shlex.quote(entry_abs)}"
    else:
        conda_sh = resolve_conda_sh(args.conda_sh)
        activate = (f"source {shlex.quote(conda_sh)} && "
                    f"conda activate {shlex.quote(conda_env)}")
        if runner.launcher == "torchrun":
            nproc = args.nproc or runner.nproc
            launcher_prefix = (f"torchrun --standalone --nproc_per_node={nproc} "
                               f"{shlex.quote(entry_abs)}")
        else:
            launcher_prefix = f"python {shlex.quote(entry_abs)}"
        launcher_prefix = f"{activate} && {launcher_prefix}"

    quoted_argv = " ".join(shlex.quote(a) for a in argv)
    cd = f"cd {shlex.quote(workdir_abs)} && "
    return f"{cd}{launcher_prefix} {quoted_argv}".rstrip()


# --------------------------------------------------------------------------- #
# Post-generation evaluation (runs in the separate `eval` conda env)
# --------------------------------------------------------------------------- #
EVAL_ENTRY = "evaluate/evaluate.py"
EVAL_DEFAULT_PROMPT_FILE = "assets/prompts/DrawBench200.txt"


def build_eval_shell_command(test_folder: str, prompt_file: str,
                             reference_folder: Optional[str], eval_out: str,
                             args: argparse.Namespace) -> str:
    """Build the shell command that runs evaluate.py in the `eval` env.

    Mirrors build_shell_command's conda-activation pattern but always uses
    `python` + the eval env, runs from ROOT, and tees output to eval_out.
    """
    entry_abs = str((ROOT / EVAL_ENTRY).resolve())
    conda_env = args.eval_env or "eval"
    conda_sh = resolve_conda_sh(args.conda_sh)

    argv = ["--test_folder", test_folder, "--prompt_file", prompt_file]
    if reference_folder:
        argv += ["--reference_folder", reference_folder]
    quoted = " ".join(shlex.quote(a) for a in argv)

    activate = f"source {shlex.quote(conda_sh)} && conda activate {shlex.quote(conda_env)}"
    mkdir = f"mkdir -p {shlex.quote(str(Path(eval_out).parent))}"
    run = (f"python {shlex.quote(entry_abs)} {quoted} "
           f"2>&1 | tee {shlex.quote(eval_out)}")
    return f"{mkdir} && {activate} && {run}"


def build_gedit_eval_shell_command(runner: ModelRunner, save_dir: str,
                                   language: str, task_type: str, backbone: str,
                                   eval_out: str, args: argparse.Namespace) -> str:
    """Build the shell command that runs evaluate_gedit.py (VIEScore) for an
    editing model.

    Runs from the variant dir (``runner.eval_cwd``) so the script's
    ``from viescore import VIEScore`` resolves, and tees output to eval_out.
    GEDIT_DATASET_PATH / QWEN25VL_MODEL_PATH are injected by run_eval via
    ctx["env"], not here.
    """
    entry_abs = str((ROOT / runner.eval_entry).resolve())
    cwd_abs = _abs(runner.eval_cwd or runner.workdir)
    conda_env = args.eval_env or "eval"
    conda_sh = resolve_conda_sh(args.conda_sh)

    argv = ["--save_dir", save_dir,
            "--instruction_language", language,
            "--task_type", task_type,
            "--backbone", backbone]
    quoted = " ".join(shlex.quote(a) for a in argv)

    activate = f"source {shlex.quote(conda_sh)} && conda activate {shlex.quote(conda_env)}"
    mkdir = f"mkdir -p {shlex.quote(str(Path(eval_out).parent))}"
    run = (f"python {shlex.quote(entry_abs)} {quoted} "
           f"2>&1 | tee {shlex.quote(eval_out)}")
    return f"cd {shlex.quote(cwd_abs)} && {mkdir} && {activate} && {run}"


def build_eval_context(runner: ModelRunner, params: Dict[str, Any],
                       args: argparse.Namespace) -> Dict[str, Any]:
    """Resolve everything the eval step needs into a single dict."""
    supported = runner.output_ext in ("png", "jpg")
    ctx: Dict[str, Any] = {
        "supported": supported,
        "eval_kind": runner.eval_kind,
        "eval_env": args.eval_env or "eval",
        "env": {},  # extra env vars injected by run_eval (gedit: dataset + backbone)
        "test_folder": params.get("outdir"),
        "reference_folder": None,
        "ref_source": "none",
        "prompt_file": None,
        "eval_out": None,
        "cmd": None,
        "__temp_eval_prompt": None,
    }
    if not supported:
        return ctx

    # GEdit-Bench editing eval: evaluate_gedit.py (VIEScore). No prompt file or
    # reference folder — VIEScore pulls originals + instructions from the dataset.
    if runner.eval_kind == "gedit":
        save_dir = params.get("outdir")
        eval_out = args.eval_out or str(Path(save_dir) / "evaluation_results.txt")
        ctx["eval_out"] = eval_out
        ctx["env"] = {
            "GEDIT_DATASET_PATH": params.get("dataset_path") or DEFAULT_GEDIT_DATASET,
            "QWEN25VL_MODEL_PATH": args.qwen25vl_model_path or DEFAULT_QWEN25VL_MODEL,
        }
        ctx["cmd"] = build_gedit_eval_shell_command(
            runner, save_dir, args.gedit_language, args.gedit_task_type,
            args.gedit_backbone, eval_out, args)
        return ctx

    ref, ref_source = resolve_eval_reference(runner, params, args)
    ctx["reference_folder"] = ref
    ctx["ref_source"] = ref_source

    # Prompt file: explicit > generation prompt_file > single --prompt (temp) >
    # DrawBench200 fallback.
    prompt_file = args.eval_prompt_file or params.get("prompt_file")
    if not prompt_file and params.get("prompt"):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                          encoding="utf-8")
        tmp.write(str(params["prompt"]) + "\n")
        tmp.close()
        prompt_file = tmp.name
        ctx["__temp_eval_prompt"] = tmp.name
    if not prompt_file:
        prompt_file = _abs(EVAL_DEFAULT_PROMPT_FILE)
    ctx["prompt_file"] = prompt_file

    eval_out = args.eval_out or str(Path(ctx["test_folder"]) / "evaluation_results.txt")
    ctx["eval_out"] = eval_out

    ctx["cmd"] = build_eval_shell_command(
        ctx["test_folder"], ctx["prompt_file"],
        ctx["reference_folder"], ctx["eval_out"], args)
    return ctx


def run_eval(ctx: Dict[str, Any], args: argparse.Namespace) -> int:
    """Execute the eval command in a fresh subprocess with the eval GPU."""
    run_env = dict(os.environ)
    gpu = args.eval_gpu if args.eval_gpu is not None else args.gpu
    if gpu is not None:
        run_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    run_env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    run_env.setdefault("XDG_CACHE_HOME", "/data/public/.cache")
    run_env.update(ctx.get("env", {}))  # gedit: GEDIT_DATASET_PATH / QWEN25VL_MODEL_PATH
    proc = subprocess.run(["bash", "-c", ctx["cmd"]], env=run_env, cwd=str(ROOT))
    return proc.returncode


# --------------------------------------------------------------------------- #
# Validation / printing
# --------------------------------------------------------------------------- #
def validate(runner: ModelRunner, entry: str, params: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable warnings (empty if all good)."""
    warnings: List[str] = []
    entry_abs = (ROOT / runner.workdir / entry).resolve()
    if not entry_abs.exists():
        warnings.append(f"entry script not found: {entry_abs}")
    mp = params.get("model_path")
    if mp and not Path(mp).exists():
        warnings.append(f"model path not found on disk: {mp} "
                        f"(set --model_path / MODEL_PATH_{runner.name.upper()})")
    return warnings


def print_list(task_filter: Optional[str]) -> None:
    order = [TaskType.IMAGE_GEN, TaskType.IMAGE_EDIT, TaskType.VIDEO_GEN]
    print(f"Cache4Diffusion — registered models (root={ROOT})\n")
    for task in order:
        if task_filter and task.value != task_filter:
            continue
        models = [r for r in REGISTRY.values() if r.task == task]
        print(f"━━ {task.label}  [{task.value}] ━━")
        if not models:
            print("    (none)")
        for r in models:
            print(f"  • {r.name:<14} env={r.conda_env:<10} out=.{r.output_ext}")
            print(f"      {r.description}")
            entry = r.entry_batch or r.entry_single
            print(f"      entry : {r.workdir}/{entry}")
            print(f"      launch: {r.launcher}" + (f" (nproc={r.nproc})" if r.launcher == 'torchrun' else ""))
            if r.note:
                print(f"      note  : {r.note}")
        print()
    print("Run a model:  python run.py --model <name> [common params] [--gpu 0] [--dry_run]")
    print("Example:      python run.py --model flux_diffusers --prompt 'a cat' --steps 50 --dry_run")


def print_plan(runner: ModelRunner, entry: str, mode: str,
               env: Dict[str, str], shell_cmd: str,
               warnings: List[str], conda_env: str) -> None:
    print(f"▶ model : {runner.name}  ({runner.task.label})")
    print(f"  mode  : {mode}")
    print(f"  entry : {runner.workdir}/{entry}")
    print(f"  conda : {conda_env}")
    print(f"  output: .{runner.output_ext}")
    print(f"\n  $ {shell_cmd}\n")
    if env:
        print("  environment:")
        for k in sorted(env):
            print(f"    {k}={env[k]}")
        print()
    if warnings:
        print("  ⚠ warnings:")
        for w in warnings:
            print(f"    - {w}")
        print()


def print_eval_plan(ctx: Dict[str, Any]) -> None:
    """Print the post-generation eval plan (shown whenever --eval is given)."""
    print(f"▶ eval  : {ctx['eval_env']} env")
    if not ctx.get("supported"):
        print("  SKIPPED: model output is not still images (evaluate.py needs png/jpg).")
        print()
        return
    if ctx.get("eval_kind") == "gedit":
        print("  kind  : GEdit-Bench / VIEScore (evaluate_gedit.py)")
        print(f"  save  : {ctx['test_folder']}")
        print(f"  out   : {ctx['eval_out']}")
        print(f"\n  $ {ctx['cmd']}\n")
        return
    print(f"  test  : {ctx['test_folder']}")
    ref = ctx.get("reference_folder")
    src = ctx.get("ref_source", "none")
    if ref:
        tag = " (auto-discovered baseline)" if src == "auto" else " (explicit)"
        print(f"  ref   : {ref}{tag}  → CLIP/ImageReward + PSNR/SSIM/LPIPS")
    else:
        print(f"  ref   : (none)  → CLIP/ImageReward only")
    print(f"  prompt: {ctx['prompt_file']}")
    print(f"  out   : {ctx['eval_out']}")
    print(f"\n  $ {ctx['cmd']}\n")


def print_benchmark_plan(params: Dict[str, Any]) -> None:
    """Print the speed-benchmark plan (shown whenever --benchmark is given)."""
    print("▶ bench : latency + FLOPs on a single generation")
    print(f"  warmup: {params.get('benchmark_warmup', 1)} untimed run(s)")
    print(f"  runs  : {params.get('benchmark_runs', 1)} timed run(s) → mean latency")
    print(f"  flops : transformer forward, summed over all steps (calflops, separate pass)")
    print(f"  report: {params.get('benchmark_report')}")
    print()



# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print_list(args.task)
        return 0

    if not args.model:
        parser.error("--model is required (use --list to see options).")
    runner = REGISTRY.get(args.model)
    if runner is None:
        parser.error(f"unknown model '{args.model}'. Available: {', '.join(sorted(REGISTRY))}.")

    if args.task and args.task != runner.task.value:
        parser.error(f"model '{runner.name}' is task '{runner.task.value}', not '{args.task}'.")

    # Editing models run over the GEdit-Bench dataset, which has a per-runner
    # default, so they need no explicit --prompt / --prompt_file / --dataset_path.
    needs_input = not (args.prompt or args.prompt_file or args.dataset_path)
    if needs_input and not runner.defaults.get("dataset_path"):
        parser.error("provide --prompt, --prompt_file, or --dataset_path.")

    # Benchmark sanity checks. --benchmark measures a single generation's
    # latency + FLOPs, so it forces single mode and is incompatible with --eval
    # (which needs saved images to score).
    if args.benchmark:
        if not runner.supports_benchmark:
            parser.error(f"model '{runner.name}' does not support --benchmark yet "
                         f"(its single entry script has no --benchmark hook).")
        if args.eval:
            parser.error("--benchmark and --eval are mutually exclusive.")
        if not args.prompt:
            parser.error("--benchmark requires a single --prompt (one generation).")

    mode = "single" if args.benchmark else resolve_mode(args)
    entry = resolve_entry(runner, mode)
    params = resolve_params(args, runner)
    _maybe_temp_prompt_file(runner, params)

    conda_env = args.conda_env or runner.conda_env

    # Resolve outdir: an explicit --outdir is used literally; otherwise derive
    # the unified layout  {outdir_root}/{model}/[{variant}/]{method}/{config}/
    if args.outdir:
        params["outdir"] = args.outdir
    else:
        params["outdir"] = resolve_unified_outdir(runner, params, args)

    # Benchmark report path defaults to <outdir>/benchmark.txt.
    if params.get("benchmark"):
        params["benchmark_report"] = (args.benchmark_report
                                      or str(Path(params["outdir"]) / "benchmark.txt"))

    # Absolutize path-like params so they survive the `cd` into workdir.
    for key in ("prompt_file", "input_image", "mask_path", "dataset_path", "outdir"):
        if params.get(key) and not str(params[key]).startswith("__"):
            params[key] = _abs(str(params[key]))

    built_argv = build_argv(runner, params)
    env = build_env(runner, params, args)
    shell_cmd = build_shell_command(runner, entry, built_argv, conda_env, args)

    warnings = [] if args.no_validate else validate(runner, entry, params)

    print_plan(runner, entry, mode, env, shell_cmd, warnings, conda_env)

    if params.get("benchmark"):
        print_benchmark_plan(params)

    eval_ctx = None
    if args.eval:
        eval_ctx = build_eval_context(runner, params, args)
        print_eval_plan(eval_ctx)

    if args.dry_run:
        return 0

    if warnings:
        print("[run.py] refusing to run because validation produced warnings.")
        print("        Fix the paths above or re-run with --no_validate to force.")
        return 2

    # Execute generation.  Pass our computed env on top of the current
    # environment; the shell activates conda (adding LD_LIBRARY_PATH etc.)
    # before the interpreter.
    run_env = dict(os.environ)
    run_env.update(env)
    proc = subprocess.run(["bash", "-c", shell_cmd], env=run_env,
                          cwd=str(ROOT))
    gen_rc = proc.returncode

    # Optional post-generation evaluation in the `eval` conda env.
    eval_rc: Optional[int] = None
    if eval_ctx is not None:
        if not eval_ctx["supported"]:
            print("[run.py] --eval skipped: model output is not still images.")
        elif gen_rc != 0:
            print(f"[run.py] generation failed (rc={gen_rc}); skipping --eval.")
        else:
            eval_rc = run_eval(eval_ctx, args)

    # Clean up temp prompt files (generation + any single-prompt eval file)
    # only after both generation and evaluation are done.
    for tmp in (params.get("__temp_prompt_file"),
                (eval_ctx or {}).get("__temp_eval_prompt")):
        if tmp and Path(tmp).exists():
            try:
                Path(tmp).unlink()
            except OSError:
                pass

    return eval_rc if eval_rc is not None else gen_rc


if __name__ == "__main__":
    sys.exit(main())
