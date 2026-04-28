# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research benchmark comparing attention cache acceleration methods for diffusion image generation. Contains four implementations targeting SDXL-UNet and Qwen-Image (DiT) models:

- **stable_diffusion_xl/** — SDXL baseline (no cache). Reference for FLOPs/quality comparisons.
- **taylorseer/** — TaylorSeer method: Taylor expansion-based feature caching for UNet blocks.
- **freqca/** — FreqCA (ours): extends TaylorSeer with frequency-domain decomposition (FFT/DCT), Hermite polynomials, and Z-cache (forecast+merge) for more aggressive step skipping.
- **qwen_fastercache/** — FasterCache adaptation for Qwen's dual-stream DiT (replaces O(n²) joint attention with cached lookups).

All SDXL subprojects share the same base model (`stabilityai/stable-diffusion-xl-base-1.0`, fp16) and evaluation pipeline (DrawBench200 prompts).

### VAE fp16 Fix

SDXL's default VAE produces `RuntimeError: expected scalar type Half but found Float` when decoding in fp16 (GroupNorm dtype mismatch). **All SDXL subprojects must use `madebyollin/sdxl-vae-fp16-fix`** instead of the default VAE:

```python
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(model_path, vae=vae, torch_dtype=torch.float16)
```

Set `SDXL_VAE_PATH` env var to use a local copy instead of downloading from HuggingFace Hub.

## Environment Setup

```bash
conda create -n stablediffusion python=3.10
conda activate stablediffusion
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install diffusers --upgrade invisible_watermark transformers accelerate safetensors
# Evaluation metrics
pip install opencv-python lpips scikit-image image-reward torch-dct
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/chengzegang/calculate-flops.pytorch.git
pip install transformers==4.55.4
```

Required HuggingFace models: `stabilityai/stable-diffusion-xl-base-1.0`, `madebyollin/sdxl-vae-fp16-fix`, `zai-org/ImageReward`, `laion/CLIP-ViT-g-14-laion2B-s12B-b42K`, `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`, `yuvalkirstain/PickScore_v1`.

## Running

All sampling uses `torchrun` for DDP. Single-GPU examples:

```bash
# Baseline SDXL
cd stable_diffusion_xl
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 sample.py

# TaylorSeer (interval controls cache period)
cd taylorseer
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 sample.py --interval 6

# FreqCA (decompose_method + use_z_cache for frequency-domain caching)
cd freqca
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 sample.py --interval 7 --decompose_method FFT --use_z_cache --forecast_steps 7

# Qwen FasterCache (8-GPU example)
cd qwen_fastercache
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 sample_ddp.py \
    --model_path /path/to/Qwen-Image --fc_start_step 15 --fc_interval 2 --fc_alpha 0.3
```

FLOPs measurement: add `--test_FLOPs` flag to any `sample.py` / `sample_ddp.py`.

Evaluation (same `evaluate.py` across SDXL subprojects):
```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py --test_folder samples/N7_FFT_ZCache
```

Batch runs: each subproject has a `run.sh` / `run_infer.sh` that sweeps intervals/methods and evaluates all settings.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SDXL_MODEL_PATH` | `stabilityai/stable-diffusion-xl-base-1.0` | Local path to SDXL base model |
| `SDXL_VAE_PATH` | `madebyollin/sdxl-vae-fp16-fix` | Local path to fp16-fix VAE |

## Architecture

### Shared Pattern (SDXL subprojects)

Each subproject has: `sample.py` (entry), `evaluate.py` (metrics), `cache_functions/` (cache logic), `pipelines/` (patched diffusers pipeline).

**Cache injection via monkey-patching:** `pipe_with_cache()` in `cache_utils.py` replaces forward methods on the UNet and its sub-modules (`DownBlock2D`, `CrossAttnDownBlock2D`, `UNetMidBlock2DCrossAttn`, `CrossAttnUpBlock2D`, `UpBlock2D`, `ResnetBlock2D`, `Transformer2DModel`, `BasicTransformerBlock`) using `types.MethodType`. This lets cache hooks live in local `models/` copies without modifying diffusers.

TaylorSeer additionally replaces all UNet sub-block forwards; FreqCA only replaces `UNet2DConditionModel.forward` (the rest are inherited from the diffusers originals).

### Cache Flow

1. `cache_init(**kwargs)` — creates `cache_dic` (config + storage) and `current` (step state machine)
2. `cal_type(cache_dic, current)` — decides whether this step is `'full'` (compute) or `'cache'` (skip)
3. On full steps: `derivative_approximation()` stores feature + finite-difference derivatives
4. On cache steps: `cache_step()` or `cache_step_merge()` reconstructs features via Taylor/Hermite extrapolation

### Key Parameters

| Param | TaylorSeer | FreqCA | Qwen FasterCache |
|-------|-----------|--------|-----------------|
| Step skipping | `--interval` | `--interval` | `--fc_interval` |
| Expansion order | `--max_order` (Taylor) | `--max_order` (Hermite/Taylor) | N/A (linear extrapolation) |
| Forecast method | N/A | `--forecast_method` (hermite/taylor) | N/A |
| Decomposition | N/A | `--decompose_method` (None/FFT/DCT) | N/A |
| Z-cache | N/A | `--use_z_cache --forecast_steps` | N/A |
| Warm-up | `--first_enhance` | `--first_enhance` | `--fc_start_step` |
| Extrapolation | N/A | `--max_order` + `--min_order` | `--fc_alpha` |

### Evaluation Metrics

`evaluate.py` computes: CLIP Score, ImageReward, PickScore (prompt alignment), PSNR, SSIM, LPIPS (against reference). Qwen FasterCache has its own `evaluate_fastercache.py` that adds FLOPs/speedup extraction from logs.

### Qwen FasterCache Differences

Targets `QwenImagePipeline` (not diffusers SDXL pipeline). Uses `pipeline_with_fastercache()` to patch `pipe.transformer` and its `QwenImageTransformerBlock`s. Simpler cache: warm-up → periodic full-attention steps → linear extrapolation on skip steps. Uses `torch.bfloat16` instead of fp16.
