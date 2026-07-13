#!/usr/bin/env python3
"""Shared latency + FLOPs benchmark harness for Cache4Diffusion.

Every per-model entry script already knows how to *build* its pipeline and
*call* it once.  This module factors out the *measurement* so a model only has
to provide:

  * ``gen_fn``       — a zero-arg callable that runs ONE generation (no saving).
  * ``transformer``  — the DiT module whose forward we count FLOPs for
                       (``pipeline.transformer`` / ``pipe.dit`` / the raw ``model``).
                       Pass ``None`` to skip FLOPs.
  * ``report_path``  — where to write the unified ``benchmark.txt``.
  * ``meta``         — dict of run metadata (model, steps, resolution, cache knobs ...).

Recipe (mirrors ``qwen/taylorseer/benchmark_sample.py``):

  1. warmup  — N untimed generation calls (CUDA kernel autotuning, cuDNN bench).
  2. latency — M timed calls (``time.perf_counter`` + ``torch.cuda.synchronize``),
               reported as the mean. Excludes model loading; includes the full
               pipe call (text encode + denoise + VAE decode), i.e. real wall time.
  3. FLOPs   — a SEPARATE profiling call. We wrap ``transformer`` with calflops'
               ``CalFlopsPipline`` (forward hooks, not a forward wrapper) and
               start/stop around the whole generation, so FLOPs accumulate across
               every timestep (cond + uncond). calflops is lazy-imported and the
               stage degrades gracefully to ``N/A`` if the package (or a GPU) is
               missing — e.g. the ``hyv15`` env used by HunyuanVideo.

Example (from an entry script)::

    from cache4diffusion_bench import run_benchmark

    def gen_once():
        return pipeline(args.prompt, num_inference_steps=args.steps,
                        generator=torch.Generator("cpu").manual_seed(args.seed),
                        guidance_scale=args.guidance_scale,
                        height=args.height, width=args.width).images[0]

    run_benchmark(
        gen_fn=gen_once,
        transformer=pipeline.transformer,
        report_path=os.path.join(args.outdir, "benchmark.txt"),
        meta={"model": "flux_diffusers", "steps": args.steps,
              "height": args.height, "width": args.width,
              "guidance_scale": args.guidance_scale, "seed": args.seed},
        warmup=args.benchmark_warmup, runs=args.benchmark_runs,
    )
"""

from __future__ import annotations

import gc
import os
import statistics
import time
from typing import Any, Callable, Dict, Optional, Tuple


# --------------------------------------------------------------------------- #
# CUDA helpers (no-op when CUDA is unavailable, so this imports anywhere).
# --------------------------------------------------------------------------- #
def _sync() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _release() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _peak_mem_gb() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass
    return None


def _reset_peak_mem() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# FLOPs stage
# --------------------------------------------------------------------------- #
def measure_flops(transformer: Any, gen_fn: Callable[[], Any]
                  ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Run ``gen_fn`` once with calflops hooks on ``transformer``.

    Returns ``(flops, macs, params)``. Any of them is ``None`` when calflops is
    unavailable or no transformer was given.
    """
    if transformer is None:
        print("[bench] FLOPs skipped: no transformer module provided.")
        return None, None, None
    try:
        from calflops.calculate_pipline import CalFlopsPipline
    except Exception as e:  # ImportError, or a CUDA-less env
        print(f"[bench] FLOPs skipped: calflops unavailable ({e}). "
              f"Install with: pip install calflops")
        return None, None, None

    # CalFlopsPipline installs forward hooks + globally patches torch functionals
    # between start/end; FLOPs accumulate across every transformer forward call
    # (i.e. across all denoising timesteps, cond + uncond).
    # NOTE: get_total_* must be read BEFORE end_flops_calculate(), which deletes
    # the per-module ``__flops__`` attributes they sum.
    counter = CalFlopsPipline(model=transformer,
                              include_backPropagation=False,
                              compute_bp_factor=2.0)
    try:
        counter.start_flops_calculate()
        gen_fn()
        _sync()
        flops = float(counter.get_total_flops())
        macs = float(counter.get_total_macs())
        params = float(counter.get_total_params())
    finally:
        counter.end_flops_calculate()
    return flops, macs, params


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def run_benchmark(*,
                  gen_fn: Callable[[], Any],
                  transformer: Any,
                  report_path: str,
                  meta: Optional[Dict[str, Any]] = None,
                  warmup: int = 1,
                  runs: int = 1,
                  save_fn: Optional[Callable[[Any], None]] = None,
                  flops_gen_fn: Optional[Callable[[], Any]] = None,
                  title: str = "Cache4Diffusion benchmark") -> Dict[str, Any]:
    """Run warmup → latency → FLOPs and write a unified report.

    ``save_fn``, if given, is called once with the last latency-run output so the
    user gets a sanity image/video alongside the metrics.

    ``flops_gen_fn``, if given, is used INSTEAD of ``gen_fn`` for the FLOPs pass.
    Use it to skip the VAE decode (e.g. ``output_type="latent"``): we only count
    transformer FLOPs, and calflops' globally-patched functionals can break some
    VAE decoders (e.g. Qwen's) during the instrumented pass.
    """
    meta = dict(meta or {})
    meta.setdefault("warmup_runs", warmup)
    meta.setdefault("latency_runs", runs)

    # 1. Warmup (untimed) --------------------------------------------------
    for i in range(max(0, warmup)):
        print(f"[bench] warmup {i + 1}/{warmup} ...")
        gen_fn()
        _sync()
        _release()

    # 2. Latency (timed, mean over `runs`) ---------------------------------
    _reset_peak_mem()
    samples = []
    last_out = None
    for i in range(max(1, runs)):
        _sync()
        t0 = time.perf_counter()
        last_out = gen_fn()
        _sync()
        samples.append(time.perf_counter() - t0)
        _release()
    latency = statistics.mean(samples)
    peak = _peak_mem_gb()

    if save_fn is not None and last_out is not None:
        try:
            save_fn(last_out)
        except Exception as e:  # don't let a save error mask the metrics
            print(f"[bench] save_fn failed: {e}")

    # 3. FLOPs (separate profiling run) ------------------------------------
    _release()
    flops, macs, params = measure_flops(transformer, flops_gen_fn or gen_fn)

    result: Dict[str, Any] = {
        "latency_sec": latency,
        "latency_samples_sec": samples,
        "peak_gpu_memory_gb": peak,
        "flops": flops,
        "macs": macs,
        "params": params,
    }
    _write_report(report_path, title, meta, result)
    _print_summary(meta, result)
    return result


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(value: Optional[float], scale: float = 1.0, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{value * scale:.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def _write_report(path: str, title: str, meta: Dict[str, Any],
                  result: Dict[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)

    lines = [title, ""]
    for key, value in meta.items():
        lines.append(f"{key}: {value}")
    lines += [
        "",
        f"latency_sec: {_fmt(result['latency_sec'], 1, 6)}",
        "latency_samples_sec: "
        + ",".join(f"{s:.6f}" for s in result["latency_samples_sec"]),
        f"peak_gpu_memory_gb: {_fmt(result['peak_gpu_memory_gb'])}",
        f"flops_T: {_fmt(result['flops'], 1e-12, 4)}",
        f"macs_T: {_fmt(result['macs'], 1e-12, 4)}",
        f"params_G: {_fmt(result['params'], 1e-9, 4)}",
        "",
        "latency_scope: single generation call (wall clock); excludes model "
        "loading; includes text encode + denoise + VAE decode",
        "flops_scope: transformer/DiT forward only; summed over all timesteps "
        "(cond + uncond); measured in a separate profiling run via calflops",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[bench] report written to: {path}")


def _print_summary(meta: Dict[str, Any], result: Dict[str, Any]) -> None:
    bar = "=" * 64
    print(bar)
    print(f"[bench] {meta.get('model', 'model')} benchmark summary")
    print(f"  latency   : {_fmt(result['latency_sec'], 1, 4)} sec "
          f"(runs={len(result['latency_samples_sec'])})")
    print(f"  flops     : {_fmt(result['flops'], 1e-12, 2)} TFLOPs")
    print(f"  macs      : {_fmt(result['macs'], 1e-12, 2)} TMACs")
    print(f"  params    : {_fmt(result['params'], 1e-9, 3)} G")
    print(f"  peak mem  : {_fmt(result['peak_gpu_memory_gb'], 1, 3)} GB")
    print(bar)


if __name__ == "__main__":
    raise SystemExit(
        "This is a library. Call run_benchmark(...) from a model entry script, "
        "or use:  python run.py --model <name> --benchmark --prompt '...' --steps N")
