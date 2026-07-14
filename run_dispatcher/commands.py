"""Build model subprocess arguments, environment, and shell commands."""

from __future__ import annotations

import argparse
import os
import shlex
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from .model_config import CACHE_KEYS, ModelRunner, ROOT
from .resolution import _abs, resolve_conda_sh

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
            # Baseline run: skip cache knobs, unless this runner declares
            # parametric baseline values (e.g. raw FLUX needs interval=1).
            if key in runner.no_cache_baseline:
                argv += [f"--{arg}", str(runner.no_cache_baseline[key])]
            continue
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
            if key in runner.no_cache_baseline:
                env[var] = str(runner.no_cache_baseline[key])
            continue  # baseline run: skip cache knobs unless parametric baseline
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
    # env.setdefault("XDG_CACHE_HOME", "/data/public/.cache")
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

