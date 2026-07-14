"""Resolve CLI values into model parameters and output paths."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .model_config import CACHE_KEYS, DEFAULT_MODEL_PATHS, ModelRunner, ROOT

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

