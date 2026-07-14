"""Validation and human-readable dispatcher output."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .model_config import ModelRunner, REGISTRY, ROOT, TaskType

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

