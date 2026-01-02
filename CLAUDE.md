# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cache4Diffusion is a unified framework for integrating feature caching-based diffusion acceleration schemes. The project implements TaylorSeer, a caching mechanism that speeds up diffusion model inference by caching and reusing computations across timesteps.

## Architecture

The project is organized into three main model directories:

### 1. Flux (`flux/`)
- Implements TaylorSeer for FLUX.1-dev models
- Key files:
  - `flux/taylorseer_flux/diffusers_taylorseer_flux.py` - Main inference script
  - `flux/taylorseer_flux/batch_infer.py` - Batch processing
  - `flux/taylorseer_flux/cache_functions/` - Caching logic
  - `flux/taylorseer_flux/forwards/` - Forward pass overrides

### 2. QWen Image (`qwen_image/`)
- Implements TaylorSeer for Qwen-Image models
- Key files:
  - `qwen_image/taylorseer_qwen_image/diffusers_taylorseer_qwen_image.py` - Main inference script
  - `qwen_image/taylorseer_qwen_image/cache_functions/` - Caching logic
  - `qwen_image/taylorseer_qwen_image/forwards/` - Forward pass overrides
  - `qwen_image/eval/` - Evaluation scripts (Geneval benchmark)

### 3. Hunyuan Video (`hunyuan_video/`)
- Implementation for Hunyuan Video models (under development)

## Common Development Tasks

### Running Single Image Inference

**For FLUX:**
```bash
cd flux
export CUDA_VISIBLE_DEVICES='5,7'  # Set your GPUs
python taylorseer_flux/diffusers_taylorseer_flux.py \
    --model /path/to/FLUX.1-dev \
    --steps 50 \
    --seed 42 \
    --dtype float16 \
    --guidance_scale 7.5 \
    --outdir outputs \
    --prompt "Your prompt here"
```

**For QWen Image:**
```bash
export PYTHONPATH="/data/huangkailin-20250908/Cache4Diffusion:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES='0,1'
python qwen_image/taylorseer_qwen_image/diffusers_taylorseer_qwen_image.py \
    --model /path/to/Qwen-Image \
    --steps 50 \
    --seed 42 \
    --dtype bfloat16 \
    --true_cfg_scale 7.5 \
    --outdir outputs \
    --prompt "Your prompt here" \
    --use_taylor
```

### Batch Inference

**FLUX Batch Processing:**
```bash
cd flux
# Set smoothing parameters
export USE_SMOOTHING="True"
export SMOOTHING_METHOD="exponential"
export SMOOTHING_ALPHA="0.9"

python taylorseer_flux/batch_infer.py \
    --model /path/to/FLUX.1-dev \
    --steps 50 \
    --prompt_file /path/to/prompts.txt \
    --outdir outputs
```

### Running Evaluations

**Geneval Evaluation for QWen Image:**
```bash
export CUDA_VISIBLE_DEVICES='7'
python qwen_image/eval/qwen_geneval.py \
    --metadata_file /path/to/evaluation_metadata.jsonl \
    --model /path/to/Qwen-Image \
    --outdir /path/to/outputs \
    --n_samples 1 \
    --steps 50 \
    --scale 7.5 \
    --dtype bfloat16
```

## Key Configuration Options

### TaylorSeer Caching Configuration
Set via environment variables:

- `USE_SMOOTHING`: Enable smoothing of cached values (True/False)
- `USE_HYBRID_SMOOTHING`: Use hybrid smoothing approach (True/False)
- `SMOOTHING_METHOD`: Smoothing method ("exponential" or "moving_average")
- `SMOOTHING_ALPHA`: Alpha parameter for exponential smoothing (0.0-1.0)

### Debug Options
- `TS_DEBUG_SHAPES`: Enable shape debugging
- `TS_STRICT_SHAPES`: Enable strict shape checking

## Model Path Configuration

Update model paths in shell scripts:
- FLUX models: `/data/public/models/FLUX.1-dev`
- QWen Image models: `/data/public/models/Qwen/Qwen-Image`

## Code Architecture Details

### TaylorSeer Mechanism
The framework implements a sophisticated caching system that:
1. Stores intermediate activations across timesteps
2. Applies Taylor series approximations for cached computations
3. Implements smoothing for temporal coherence
4. Supports different caching strategies (ToCa, random, etc.)

### Forward Pass Overrides
Each model implements custom forward methods that:
- Intercept standard diffusion model forward passes
- Apply caching logic at different layers
- Support both single and double transformer blocks
- Handle attention map caching

### Cache Structure
The cache system maintains:
- `cache[-1]` and `cache[-2]`: Current and previous timestep caches
- Separate caches for double/single stream processing
- Attention maps for different transformer components
- Cache indices for efficient lookup

## Development Notes

- The project heavily modifies diffusers pipeline behavior through method overrides
- Cache initialization happens during the first forward pass
- The framework supports both single-GPU and multi-GPU inference
- Memory optimization is critical due to the caching overhead