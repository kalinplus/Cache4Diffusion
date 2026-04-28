# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TaylorSeer is an ICCV 2025 research implementation that accelerates diffusion models (DiT) using Taylor series expansion for feature forecasting. It achieves ~5x lossless computational compression on FLUX.1-dev and HunyuanVideo without additional training.

## Repository Structure

This is a **monorepo of independent model implementations** — each subdirectory is a self-contained codebase with its own dependencies, entry points, and model weights:

| Directory | Model | Entry Points |
|-----------|-------|-------------|
| `TaylorSeer-FLUX` | FLUX image generation | `python -m flux`, `src/sample.py`, `demo_*.py` |
| `TaylorSeer-HunyuanVideo` | HunyuanVideo video gen | `sample_video.py`, `gradio_server.py` |
| `TaylorSeer-Wan2.1` | Wan2.1 video generation | `generate.py` |
| `TaylorSeer-DiT` | DiT (Diffusion Transformer) | `sample.py`, `sample_ddp.py` |
| `TaylorSeer-HiDream` | HiDream image generation | `src/` scripts |
| `TaylorSeer-HiDream-I1-nf4` | HiDream (NF4 quantized) | `hdi1/` package |
| `TaylorSeer-FramePack` | FramePack video interpolation | Diffusers-based |
| `TaylorSeer-FLUX-Kontext` | FLUX + Kontext | Same as FLUX |
| `TaylorSeers-Diffusers` | Diffusers integration | Inference scripts for FLUX |
| `TaylorSeers-xDiT` | xDiT multi-GPU parallel | Inference scripts for FLUX |

## Core Algorithm Architecture

Each model implementation follows the same pattern of three injected subsystems:

1. **`taylor_utils/`** — The core Taylor series math:
   - `derivative_approximation()` — computes higher-order derivatives via finite differences
   - `taylor_formula()` — predicts features at future timesteps using Taylor expansion
   - `taylor_cache_init()` — initializes per-layer/per-stream Taylor coefficient storage

2. **`cache_functions/`** — Cache management and scheduling:
   - `cache_init()` — allocates cache dictionaries keyed by stream/layer/module; sets mode (`'Taylor'`, `'ToCa'`, `'original'`, `'Delta'`) and key parameters: `fresh_threshold`, `max_order`, `first_enhance`
   - `cal_type()` — determines whether each denoising step computes `'full'`, `'taylor_cache'`, `'ToCa'`, or `'Delta-Cache'`
   - `force_scheduler.py` — schedules when cached features must be refreshed
   - `fresh_ratio_scheduler.py` — adjusts fresh ratio over timesteps

3. **`forwards/`** — Modified forward passes that integrate caching into the transformer layers (present in TaylorSeers-Diffusers, TaylorSeers-xDiT, TaylorSeer-Wan2.1)

The algorithm alternates between full computation steps (which update Taylor coefficients) and cached steps (which predict features via Taylor expansion instead of running the full model).

## Key Configuration Parameters

Set in `cache_functions/cache_init.py` per implementation:
- **`mode`**: `'Taylor'` (TaylorSeer), `'ToCa'` (reuse baseline), `'original'`, `'Delta'`
- **`fresh_threshold`**: steps between full computations (controls acceleration ratio — higher = faster)
- **`max_order`**: Taylor expansion order (default 1; higher orders improve accuracy at large intervals)
- **`first_enhance`**: number of initial steps to run fully before caching kicks in (default 3)
- **`cal_threshold`**: runtime threshold computed from `fresh_threshold` by the scheduler

## Installation & Running

No monorepo-wide install. Each subdirectory manages its own dependencies:

```bash
# TaylorSeer-FLUX (pip-installable)
cd TaylorSeer-FLUX && pip install -e .

# Others (requirements.txt)
cd TaylorSeer-HunyuanVideo && pip install -r requirements.txt
cd TaylorSeer-Wan2.1 && pip install -r requirements.txt
```

**Multi-GPU** uses `torchrun`:
```bash
torchrun --nproc_per_node=8 sample_video.py ...    # HunyuanVideo
torchrun --nproc_per_node=8 generate.py --dit_fsdp  # Wan2.1
torchrun --nproc_per_node=8 sample_ddp.py ...       # DiT
```

## Linting & Formatting

Only TaylorSeer-FLUX has linting configured (via `ruff` in `pyproject.toml`):
- Line length: 110, Python 3.10+, double quotes, space indentation
- `ruff check . && ruff format .`

## Testing

Minimal formal test infrastructure. Only `TaylorSeer-HunyuanVideo/tests/test_attention.py` exists (distributed attention correctness via torchrun). Testing is primarily manual inference validation.

## Dependencies

Core stack: PyTorch (>=2.4), diffusers (>=0.31), transformers (>=4.46), accelerate, einops, flash-attn, safetensors. Video models additionally need opencv-python, imageio, imageio-ffmpeg. Wan2.1 requires dashscope for prompt extension.

## License

GNU GPL v3
