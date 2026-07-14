"""Model runner types, defaults, and the model registry."""

from __future__ import annotations

import dataclasses
import enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent

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

    # For runners whose caching is purely PARAMETRIC (no on/off flag — e.g. the
    # raw FLUX repo, where caching is always via denoise_cache(interval/order/...)),
    # --no_cache cannot "skip" caching; it needs baseline cache values instead.
    # This maps common cache keys to the values injected (rather than skipped)
    # under --no_cache.  e.g. {"cache_interval": 1} → every step fresh = baseline.
    no_cache_baseline: Dict[str, Any] = dataclasses.field(default_factory=dict)

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
    "flux_schnell": "/mnt/workspace/hkl/models/black-forest-labs/FLUX.1-schnell",
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
        env_map={"smoothing_alpha": "SMOOTHING_ALPHA",
                 "smoothing_method": "SMOOTHING_METHOD"},
        env_bool_map={"use_smoothing": "USE_SMOOTHING"},
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
             "single --prompt is written to a temp prompt file; smoothing is "
             "forwarded through USE_SMOOTHING/SMOOTHING_METHOD/SMOOTHING_ALPHA. "
             "Caching is parametric, so --no_cache injects interval=1 as the "
             "true baseline.",
        no_cache_baseline={"cache_interval": 1},
    ),

    "flux_schnell": ModelRunner(
        name="flux_schnell",
        task=TaskType.IMAGE_GEN,
        description="FLUX.1-schnell text-to-image (raw BFL repo + TaylorSeer) — flux/taylorseer/",
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
        env_map={"smoothing_alpha": "SMOOTHING_ALPHA",
                 "smoothing_method": "SMOOTHING_METHOD"},
        env_bool_map={"use_smoothing": "USE_SMOOTHING"},
        model_name="flux-schnell",
        model_name_arg="model_name",
        model_path_env="__flux_root__",  # placeholder; real vars come from env_builder
        env_builder=_flux_raw_env_builder("flux1-schnell.safetensors"),
        # Caching here is purely parametric (sample.py always calls denoise_cache;
        # there is no on/off flag), so --no_cache must inject a real baseline:
        # interval=1 makes every step a fresh (full) step.
        no_cache_baseline={"cache_interval": 1},
        defaults={"steps": 4, "seed": 0,
                  "width": 1024, "height": 1024, "cache_interval": 2,
                  "cache_max_order": 0, "cache_first_enhance": 1,
                  "smoothing_alpha": 0.8, "smoothing_method": "exponential"},
        output_ext="jpg",
        dtype_default="bfloat16",
        note="4-step distilled FLUX: sample.py asserts num_steps==4 and ignores "
             "guidance (guidance_embed=False), so guidance is not in the defaults. "
             "Weights via env FLUX_MODEL/FLUX_AE/T5_MODEL_PATH/CLIP_MODEL_PATH; "
             "smoothing via USE_SMOOTHING/SMOOTHING_ALPHA. Caching is parametric, so "
             "--no_cache injects interval=1 (every step fresh) as the true baseline.",
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
