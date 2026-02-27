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

## Environment

QwenImage 模型使用 `qwenimage` conda 环境。在 shell 中运行命令时使用 `conda run`：

```bash
conda run -n qwenimage python -c "import sys; print(sys.executable)"
conda run -n qwenimage pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Development Notes

- The project heavily modifies diffusers pipeline behavior through method overrides
- Cache initialization happens during the first forward pass
- The framework supports both single-GPU and multi-GPU inference
- Memory optimization is critical due to the caching overhead

---

## Refactoring: Strategy + Adapter Architecture

**Branch:** `refactor/taylorseer-core`
**Reference:** `~/Cache4Diffusion-main` (git worktree of main branch)
**Run from:** project root `/home/hkl/Cache4Diffusion/` with `PYTHONPATH=/home/hkl/Cache4Diffusion`

### Goal

Decouple N×N (models × methods) into N+N (models + methods) via Strategy Pattern + Adapter Pattern.

### New Package Structure

```
taylorseer_core/       # Shared math/scheduler/config (DONE)
  math.py              # Taylor math, cache_init(), smoothing
  scheduler.py         # cal_type(), force_scheduler()
  config.py            # TaylorSeerConfig dataclass
  forward_utils.py     # update_cache_or_approximate()

caching_core/          # Strategy layer (DONE)
  base.py              # CacheStrategy ABC
  context.py           # StepContext (dict-compatible dataclass)
  strategies/
    taylorseer_strategy.py  # TaylorSeerStrategy wrapping taylorseer_core

model_adapters/        # Adapter layer (DONE)
  base.py              # ModelAdapter ABC + create_forward_fn()
  info.py              # ModelInfo dataclass
  factory.py           # patch_model_with_cache()
  adapters/
    flux_adapter.py    # FluxAdapter (full FLUX forward logic)
```

### Key Design Decisions

- `StepContext` supports `ctx['key']` dict-style access for backward compat with `taylorseer_core`
- `FluxAdapter.create_forward_fn()` overrides the base class version — it must include full FLUX embedding preprocessing (`x_embedder`, `temb`, `pos_embed`) before the block loop
- `patch_model_with_cache()` assigns to `model.forward` (instance attr, not class), so `patched_forward` must NOT have `self` as first arg
- All imports use absolute paths from project root
- `QwenImageAdapter` uses dual `'cond'`/`'uncond'` cache branches (not `'double_stream'`) because QwenImage's true CFG calls forward twice per step. Branch name is read from diffusers' `cache_context` via wrapped `model.cache_context` → `model._ts_cache_branch`
- `cache_init()` accepts optional `branches` dict (e.g. `{'cond': 60, 'uncond': 60}`) to override the default `double_stream`/`single_stream` layout

### Migration Stages

| Stage | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Done | Extract `taylorseer_core` shared math/scheduler |
| 2 | ✅ Done | Define `CacheStrategy` ABC |
| 3 | ✅ Done | Implement `TaylorSeerStrategy` |
| 4 | ✅ Done | Define `ModelAdapter` ABC + `FluxAdapter` |
| 5 | ✅ Done | Verify FLUX end-to-end with new framework |
| 6 | ✅ Done | Add `QwenImageAdapter`, `HunyuanVideoAdapter` (both verified) |
| 7 | 🔲 Todo | Migrate ClusCa, SpeCa to strategy interface |
| 8 | 🔲 Todo | Unified factory `create_pipeline(model, method)` |

### Automated Testing

**`test.sh` — 语法/语义正确性检查**（`--steps 1`，快速冒烟测试）：

```bash
MODEL_NAME=flux bash test.sh
MODEL_NAME=qwen_image bash test.sh
```

能跑通说明代码没有语法/导入/语义错误。

**`infer.sh` — 逻辑正确性检查**（多步推理，验证输出质量）：

```bash
MODEL_NAME=qwen_image bash infer.sh
```

能跑且输出图片有大致形状（不是纯黑、纯噪声）说明代码逻辑正确。需要使用视觉能力查看输出图片或由用户反馈确认。

### Verify FLUX

```bash
bash flux/scripts/infer_taylorseer_single_flux.sh
```

### Known Issues / Pitfalls

- `cal_type()` / `force_scheduler()` in `taylorseer_core/scheduler.py` use `current['key']` dict access — `StepContext.__getitem__` handles this
- FLUX `patched_forward` must handle `pooled_projections`, `timestep`, `img_ids`, `txt_ids`, `guidance` — these are FLUX-specific, not generic transformer args
- `forward_double_block_full` returns `(encoder_hidden_states, hidden_states)` — note enc comes first (matches original diffusers convention)
