"""Build and run post-generation evaluation commands."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from .model_config import (
    DEFAULT_GEDIT_DATASET,
    DEFAULT_QWEN25VL_MODEL,
    ModelRunner,
    ROOT,
)
from .resolution import _abs, resolve_conda_sh, resolve_eval_reference

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
    # run_env.setdefault("XDG_CACHE_HOME", "/data/public/.cache")
    run_env.update(ctx.get("env", {}))  # gedit: GEDIT_DATASET_PATH / QWEN25VL_MODEL_PATH
    proc = subprocess.run(["bash", "-c", ctx["cmd"]], env=run_env, cwd=str(ROOT))
    return proc.returncode

