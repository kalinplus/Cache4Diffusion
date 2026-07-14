#!/usr/bin/env python3
"""Run the native ``flux_schnell`` TaylorSeer configuration sweep in parallel.

This is the schnell counterpart of ``flux_sweep.py``. The differences from the
dev sweep are intentional and reflect that FLUX.1-schnell runs in exactly 4
denoising steps, so the cache interval can only be 1 or 2:

  * ``--model flux_schnell``, ``--steps 4`` (schnell asserts steps == 4).
  * Config space: N(cache_interval) ∈ {1, 2} × O(cache_max_order) ∈ {0, 1}
    × F(cache_first_enhance) = 1 × smoothing ∈ {none, exponential A0.8}.
    That is 8 cached configs + 1 baseline.
  * Summary lands at ``outputs/flux_schnell/sweep_summary.tsv`` (same columns as
    the dev sweep).

The baseline is completed first so every cached run can use it as the metric
reference. Cached configurations are then assigned one at a time to the GPUs in
``--gpus``. Each worker launches the existing top-level ``run.py``; this file
only schedules subprocesses and collects their reports.

``--collect-only`` skips every run and instead scans the existing output tree
(``outputs/flux_schnell/{baseline,taylorseer}/``), parsing each config's
``benchmark.txt`` / ``evaluation_results.txt`` straight into the summary TSV.
Use it to regenerate the summary from results already on disk — including ones
produced by ``flux_schnell_sweep.sh`` — without touching a GPU.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Iterable, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = "flux_schnell"          # sub-directory under --outdir_root
RUN_MODEL = "flux_schnell"          # value passed to run.py --model
PRINT_LOCK = Lock()


@dataclass(frozen=True)
class SweepConfig:
    label: str
    slug: str
    interval: Optional[int]
    max_order: Optional[int]
    first_enhance: Optional[int]
    smoothing: str
    alpha: Optional[float]
    baseline: bool = False


@dataclass
class ConfigResult:
    config: SweepConfig
    gpu: str
    output_dir: Path
    log_path: Path
    benchmark_rc: Optional[int] = None
    generation_rc: Optional[int] = None
    status: str = "pending"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Native FLUX.1-schnell TaylorSeer sweep through run.py."
    )
    parser.add_argument(
        "--gpus", default=os.environ.get("GPUS", os.environ.get("GPU", "0")),
        help="Comma-separated physical GPU ids; one configuration per GPU (default: 0).",
    )
    parser.add_argument("--prompt_file", default="assets/prompts/DrawBench200.txt")
    parser.add_argument("--prompt", default=None, help="Use one prompt instead of --prompt_file.")
    parser.add_argument("--bench_prompt", default="a red panda wearing a top hat, photorealistic, highly detailed")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--flux_t5_root", default=None)
    parser.add_argument("--flux_clip_root", default=None)
    parser.add_argument("--outdir_root", default="outputs")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--first_enhance", type=int, default=1)
    parser.add_argument("--benchmark_warmup", type=int, default=1)
    parser.add_argument("--benchmark_runs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=None,
                        help="Maximum concurrent jobs; capped at the number of GPUs.")
    parser.add_argument("--logs_root", default=None)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--python", default=os.environ.get("RUN_PYTHON", sys.executable),
                        help="Interpreter used to invoke run.py.")
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without executing them.")
    parser.add_argument("--resume", action="store_true", help="Skip jobs with a success marker.")
    parser.add_argument("--origin-only", action="store_true", help="Run only origin.")
    parser.add_argument("--cached-only", action="store_true",
                        help="Skip origin; require the existing baseline directory.")
    parser.add_argument("--collect-only", action="store_true",
                        help="Don't run anything; build the summary TSV from existing outputs.")
    parser.add_argument("--no-benchmark", dest="benchmark", action="store_false", default=True)
    parser.add_argument("--no-eval", dest="evaluation", action="store_false", default=True)
    return parser.parse_args()


def parse_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus:
        raise SystemExit("--gpus must contain at least one GPU id, for example 0,1,2")
    if len(set(gpus)) != len(gpus):
        raise SystemExit(f"--gpus contains duplicates: {value!r}")
    return gpus


def configs(first_enhance: int) -> list[SweepConfig]:
    """8 cached configs: N ∈ {1,2} × O ∈ {0,1} × F=1 × {none, exponential A0.8}.

    Mirrors flux_schnell_sweep.sh's INTERVALS=(1 2) MAX_ORDERS=(0 1) ALPHAS=(0 0.8):
    alpha 0 → no smoothing, alpha 0.8 → exponential smoothing.
    """
    result: list[SweepConfig] = []
    for interval in (1, 2):
        for max_order in (0, 1):
            common = f"N{interval}O{max_order}F{first_enhance}"
            result.append(SweepConfig(
                label=f"{common} no_smoothing", slug=f"{common}_none",
                interval=interval, max_order=max_order, first_enhance=first_enhance,
                smoothing="none", alpha=None,
            ))
            result.append(SweepConfig(
                label=f"{common} exponential_A0.8", slug=f"{common}_expA0.8",
                interval=interval, max_order=max_order, first_enhance=first_enhance,
                smoothing="exponential", alpha=0.8,
            ))
    return result


def origin_config() -> SweepConfig:
    return SweepConfig(
        label="origin baseline", slug="origin", interval=None, max_order=None,
        first_enhance=None, smoothing="none", alpha=None, baseline=True,
    )


def alpha_text(alpha: Optional[float]) -> str:
    if alpha is None:
        return "0"
    return f"{alpha:g}"


def output_dir(args: argparse.Namespace, config: SweepConfig) -> Path:
    variant = f"/{args.variant}" if args.variant else ""
    if config.baseline:
        config_name = f"S{args.steps}"
        method = "baseline"
    else:
        config_name = (
            f"S{args.steps}_N{config.interval}O{config.max_order}F{config.first_enhance}"
        )
        if config.smoothing != "none":
            config_name += f"A{alpha_text(config.alpha)}"
        method = "taylorseer"
    base = ROOT / args.outdir_root / MODEL_DIR / method / config_name
    return Path(f"{base}/{variant.strip('/')}".rstrip("/")) if variant else base


def log_path(args: argparse.Namespace, config: SweepConfig) -> Path:
    root = Path(args.logs_root) if args.logs_root else ROOT / args.outdir_root / f"{MODEL_DIR}_sweep_logs"
    return root / f"{config.slug}.log"


def base_command(args: argparse.Namespace, gpu: str, config: SweepConfig) -> list[str]:
    command = [args.python, str(ROOT / "run.py"), "--model", RUN_MODEL, "--gpu", gpu,
               "--steps", str(args.steps), "--seed", str(args.seed),
               "--width", str(args.width), "--height", str(args.height),
               "--outdir_root", args.outdir_root]
    if args.variant:
        command += ["--variant", args.variant]
    if args.model_path:
        command += ["--model_path", args.model_path]
    if args.flux_t5_root:
        command += ["--flux_t5_root", args.flux_t5_root]
    if args.flux_clip_root:
        command += ["--flux_clip_root", args.flux_clip_root]

    if config.baseline:
        command.append("--no_cache")
    else:
        command += ["--cache_interval", str(config.interval),
                    "--cache_max_order", str(config.max_order),
                    "--cache_first_enhance", str(config.first_enhance)]
        if config.smoothing != "none":
            command += ["--use_smoothing", "--smoothing_method", config.smoothing]
            if config.alpha is not None:
                command += ["--smoothing_alpha", str(config.alpha)]
    if args.prompt is not None:
        command += ["--prompt", args.prompt]
    else:
        command += ["--prompt_file", args.prompt_file]
    return command


def command_for(args: argparse.Namespace, gpu: str, config: SweepConfig,
                kind: str) -> list[str]:
    command = base_command(args, gpu, config)
    if kind == "benchmark":
        # run.py requires --prompt for benchmark, even when generation uses a file.
        prompt_flag = "--prompt" if args.prompt is not None else "--prompt_file"
        prompt_index = command.index(prompt_flag)
        del command[prompt_index:prompt_index + 2]
        command += ["--prompt", args.bench_prompt, "--benchmark",
                    "--benchmark_warmup", str(args.benchmark_warmup),
                    "--benchmark_runs", str(args.benchmark_runs)]
    elif kind == "generation" and args.evaluation:
        command.append("--eval")
    return command


def format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in command)


def print_job(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def run_command(command: Sequence[str], log: Path, dry_run: bool) -> int:
    if dry_run:
        return 0
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"$ {format_command(command)}\n")
        stream.flush()
        completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        stream.write(f"[exit_code] {completed.returncode}\n")
        return completed.returncode


def resume_ready(args: argparse.Namespace, config: SweepConfig,
                 marker: Path, outdir: Path) -> bool:
    if not args.resume or not marker.exists():
        return False
    if args.benchmark and not (outdir / "benchmark.txt").exists():
        return False
    if args.evaluation and not config.baseline and not (outdir / "evaluation_results.txt").exists():
        return False
    return True


def run_config(args: argparse.Namespace, config: SweepConfig, gpu: str) -> ConfigResult:
    outdir = output_dir(args, config)
    log = log_path(args, config)
    result = ConfigResult(config=config, gpu=gpu, output_dir=outdir, log_path=log)
    marker = log.with_suffix(".success")

    if resume_ready(args, config, marker, outdir):
        result.status = "resumed"
        result.benchmark_rc = 0 if args.benchmark else None
        result.generation_rc = 0
        return result

    print_job(f"[{config.label}] GPU {gpu}")
    if args.benchmark:
        command = command_for(args, gpu, config, "benchmark")
        print_job(f"  benchmark: {format_command(command)}")
        result.benchmark_rc = run_command(command, log, args.dry_run)
        if result.benchmark_rc != 0:
            result.status = "failed_benchmark"
            return result

    command = command_for(args, gpu, config, "generation")
    print_job(f"  generation: {format_command(command)}")
    result.generation_rc = run_command(command, log, args.dry_run)
    if result.generation_rc != 0:
        result.status = "failed_generation"
        return result

    if not args.dry_run:
        marker.write_text("success\n", encoding="utf-8")
    result.status = "dry_run" if args.dry_run else "ok"
    return result


def metric_values(path: Path, keys: Iterable[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    wanted = set(keys)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in wanted:
            values[key] = value.strip()
    return values


def eval_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith("Result:")) + 1
    except StopIteration:
        return {}
    names = ("clip_score", "imagereward", "psnr", "ssim", "lpips")
    # Accept inf/nan too: cached configs that reproduce the baseline pixel-for-pixel
    # (e.g. cache_interval=1) report PSNR=inf, which the plain float regex would
    # silently drop and shift every following column by one.
    token = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|inf|nan)", re.IGNORECASE)
    numbers: list[str] = []
    for line in lines[start:]:
        if token.fullmatch(line.strip()):
            numbers.append(line.strip())
        if len(numbers) == len(names):
            break
    return dict(zip(names, numbers))


def summary_path(args: argparse.Namespace) -> Path:
    path = Path(args.summary) if args.summary else ROOT / args.outdir_root / MODEL_DIR / "sweep_summary.tsv"
    return path if path.is_absolute() else ROOT / path


def write_summary(args: argparse.Namespace, results: Sequence[ConfigResult]) -> Path:
    path = summary_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["label", "gpu", "status", "output_dir", "log", "benchmark_rc",
              "generation_rc", "latency_sec", "flops_T", "macs_T",
              "peak_gpu_memory_gb", "clip_score", "imagereward", "psnr", "ssim", "lpips"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for result in results:
            row = {"label": result.config.label, "gpu": result.gpu, "status": result.status,
                   "output_dir": str(result.output_dir), "log": str(result.log_path),
                   "benchmark_rc": result.benchmark_rc, "generation_rc": result.generation_rc}
            row.update(metric_values(result.output_dir / "benchmark.txt",
                                     ("latency_sec", "flops_T", "macs_T", "peak_gpu_memory_gb")))
            row.update(eval_values(result.output_dir / "evaluation_results.txt"))
            writer.writerow(row)
    return path


def print_summary(results: Sequence[ConfigResult], summary: Path) -> None:
    ok = sum(result.status in ("ok", "resumed", "dry_run", "collected") for result in results)
    failed = len(results) - ok
    print(f"\nSweep finished: {ok} succeeded, {failed} missing/failed.")
    print(f"Summary: {summary}")
    for result in results:
        print(f"  {result.status:18} GPU {result.gpu:>3} {result.config.label}")


def collect_existing(args: argparse.Namespace) -> list[ConfigResult]:
    """Scan the output tree and build results from what is already on disk.

    Enumerates the expected baseline + 8 cached configs and marks each
    ``collected`` (benchmark.txt present) or ``missing`` (dir absent / no
    benchmark). Metric/eval columns are filled in by ``write_summary``.
    """
    results: list[ConfigResult] = []
    for config in [origin_config(), *configs(args.first_enhance)]:
        outdir = output_dir(args, config)
        log = log_path(args, config)
        result = ConfigResult(config=config, gpu="-", output_dir=outdir, log_path=log)
        if (outdir / "benchmark.txt").exists():
            result.status = "collected"
            result.benchmark_rc = 0
            result.generation_rc = 0
        else:
            result.status = "missing"
        results.append(result)
    results.sort(key=lambda r: (0 if r.config.baseline else 1, r.config.slug))
    return results


def main() -> int:
    args = parse_args()

    if args.collect_only:
        results = collect_existing(args)
        summary = write_summary(args, results)
        print_summary(results, summary)
        return 0

    gpus = parse_gpus(args.gpus)
    if args.origin_only and args.cached_only:
        raise SystemExit("--origin-only and --cached-only are mutually exclusive")

    origin = origin_config()
    cached = configs(args.first_enhance)
    baseline_dir = output_dir(args, origin)
    results: list[ConfigResult] = []

    if not args.cached_only:
        print("Running origin first; cached jobs will start only after it succeeds.")
        origin_result = run_config(args, origin, gpus[0])
        results.append(origin_result)
        if origin_result.status not in ("ok", "resumed", "dry_run"):
            summary = summary_path(args)
            if not args.dry_run:
                summary = write_summary(args, results)
            print_summary(results, summary)
            return 1
    elif not baseline_dir.exists() and not args.dry_run:
        raise SystemExit(f"--cached-only requires an existing baseline directory: {baseline_dir}")

    if args.origin_only:
        summary = summary_path(args)
        if not args.dry_run:
            summary = write_summary(args, results)
        print_summary(results, summary)
        return 0

    print(f"Scheduling {len(cached)} cached configurations across GPUs: {', '.join(gpus)}")
    max_workers = min(args.workers or len(gpus), len(gpus))
    if max_workers < 1:
        raise SystemExit("--workers must be at least 1")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(run_config, args, config, gpus[index % len(gpus)]): config
            for index, config in enumerate(cached)
        }
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda result: (0 if result.config.baseline else 1, result.config.slug))
    summary = summary_path(args)
    if not args.dry_run:
        summary = write_summary(args, results)
    print_summary(results, summary)
    return 0 if all(result.status in ("ok", "resumed", "dry_run") for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
