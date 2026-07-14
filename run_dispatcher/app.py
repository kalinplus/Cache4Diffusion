"""Application orchestration for the unified inference dispatcher."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Optional, Sequence

from .commands import _maybe_temp_prompt_file, build_argv, build_env, build_shell_command
from .evaluation import build_eval_context, run_eval
from .model_config import REGISTRY, ROOT
from .parser import build_parser
from .reporting import (
    print_benchmark_plan,
    print_eval_plan,
    print_list,
    print_plan,
    validate,
)
from .resolution import (
    _abs,
    resolve_entry,
    resolve_mode,
    resolve_params,
    resolve_unified_outdir,
)


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _should_skip_existing_output(outdir: str, benchmark: bool) -> bool:
    """Return whether this run already has an output to preserve.

    A benchmark creates the output directory before the following generation
    call, so a directory containing only ``benchmark.txt`` is intentionally
    allowed through for generation. Any other existing directory is treated as
    a user-owned/incomplete result and requires manual deletion to rerun.
    """
    path = Path(outdir)
    if not path.exists():
        return False
    if not path.is_dir():
        return True

    files = [item for item in path.iterdir() if item.is_file()]
    has_images = any(item.suffix.lower() in _IMAGE_SUFFIXES for item in files)
    if benchmark:
        return has_images or (path / "benchmark.txt").exists()

    benchmark_only = files and all(item.name == "benchmark.txt" for item in files)
    return not benchmark_only


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

    if (not args.dry_run and
            _should_skip_existing_output(str(params["outdir"]), bool(params.get("benchmark")))):
        print(f"[run.py] output directory already exists; skipping: {params['outdir']}")
        return 0

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
