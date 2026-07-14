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

### Unified Output Paths & Auto-Eval (via `run.py`)

When launched through `run.py`/`run.sh` **without** `--outdir`, outputs go to a
unified, distinguishable layout (so runs of different model / method / config
never collide):

```
{outdir_root}/{model}/[{variant}/]{method}/{config}/
```

- `outdir_root` — `--outdir_root`, default `outputs`.
- `model` — model name (e.g. `flux_diffusers`, `qwen_image`).
- `variant` — optional `--variant <tag>` (e.g. `lora-animation2k_v1`, `quant-nf4`).
- `method` — `baseline` (`--no_cache`) | `taylorseer` (caching on); `--method` overrides for labelling.
- `config` — `S{steps}` +, only when caching on **and** the model forwards the knobs, `N{interval}O{order}F{first_enhance}` +, when `--use_smoothing`, `A{alpha}`. Examples: `S50_N5O1F3A0`, `S50_N5O1F3A0.8`, baseline `S50`.

> **Default `cache_first_enhance` (F)**: for 50-step runs — the common case across FLUX/QWen/etc. — `cache_first_enhance = 3` for **all** methods (hence the `F3` in names like `S50_N5O1F3`). `flux_sweep.py`/`flux_sweep.sh` default to 3; the only exception is FLUX.1-schnell, which uses `F1` because it runs just 4 steps (`S4_N2O1F1`).

An explicit `--outdir <dir>` overrides the whole layout (used literally).

`--eval` runs `evaluate/evaluate.py` in the separate `eval` conda env right
after a successful still-image generation: CLIP + ImageReward always, and
PSNR/SSIM/LPIPS when a reference is available. The reference auto-resolves to
the sibling `baseline/S{steps}/` folder if it exists; override with
`--eval_reference_folder`. Metrics land at `<config>/evaluation_results.txt`.

```bash
# generate into the unified layout, then auto-eval (5 metrics if baseline exists)
python run.py --model flux_diffusers --prompt "a cat" --steps 50 --eval --gpu 0

# a cached + smoothed run, tagged with a LoRA variant
python run.py --model flux_diffusers --prompt "a cat" --steps 50 \
    --variant lora-animation2k_v1 --use_smoothing --smoothing_alpha 0.8 --eval
# → outputs/flux_diffusers/lora-animation2k_v1/taylorseer/S50_N5O1F1A0.8/
```

### Speed benchmark — latency + FLOPs (via `run.py --benchmark`)

`--benchmark` runs a SINGLE generation and measures real wall-clock latency
(after warmup) plus total transformer/DiT FLOPs (via `calflops`, in a separate
profiling pass). Report lands at `<config>/benchmark.txt`. Forces single mode
and is incompatible with `--eval`. Knobs: `--benchmark_warmup N` (untimed
runs, default 1), `--benchmark_runs N` (timed runs, report the mean, default 1).

```bash
# baseline (no cache) latency + FLOPs
python run.py --model flux_diffusers --prompt "a cat" --steps 50 --benchmark --gpu 0

# cached (TaylorSeer) — compare against the baseline run
python run.py --model flux_diffusers --prompt "a cat" --steps 50 --benchmark --gpu 0 --no_cache
# → add --no_cache for the baseline; otherwise caching knobs (N/O/F) apply
```

What is measured:
- `latency_sec` — one generation call (perf_counter + cuda.synchronize);
  excludes model loading, includes text encode + denoise + VAE decode.
- `flops_T` / `macs_T` — transformer forward only, **summed over all timesteps**
  (cond + uncond); measured in a separate pass so the calflops instrumentation
  does not pollute latency. For diffusers models the FLOPs pass uses
  `output_type="latent"` (skips the VAE) to avoid a calflops×VAE clash.
- `params_G` — transformer parameter count; `peak_gpu_memory_gb`.

Caveats:
- All 8 registered models support `--benchmark`. The shared harness lives at
  `cache4diffusion_bench.py` (imported by each entry script).
- `calflops` is installed in `infer`/`eval` but **not** `hyv15` (HunyuanVideo):
  latency still works there; FLOPs report `N/A` until `pip install calflops`.
- `qwen_image` caching at native 1328² needs >80 GB (single-GPU OOM); use
  `--no_cache` or a smaller resolution, or run on multi-GPU `device_map`.

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

生成模型统一首先尝试使用 `infer` conda 环境（如果不行，可以继续尝试`hyv15`环境，都不行则上报）。在 shell 中运行命令时使用 `conda run`：

```bash
conda run -n infer python -c "import sys; print(sys.executable)"
conda run -n infer pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Development Notes

- The project heavily modifies diffusers pipeline behavior through method overrides
- Cache initialization happens during the first forward pass
- The framework supports both single-GPU and multi-GPU inference
- Memory optimization is critical due to the caching overhead

## Testing & Debugging(测试与调试建议)

When fixing bugs or implementing features, Claude Code **should actually run the code on GPU to verify it works end-to-end**, rather than only editing and assuming it is correct. 修复或实现功能后,鼓励真实跑一遍确认能跑通。

- **先查可用显存**:运行 `nvidia-smi` 查看各 GPU 的显存占用,挑空闲的卡(FLUX/QWen 等模型单卡通常需要 ≥20GB 空闲显存)。
- **自己选卡跑通验证**:设置 `CUDA_VISIBLE_DEVICES=<空闲卡号>`,用 `infer` conda 环境实际运行(见上文 Environment 一节,用 `conda run -n infer`),确保改动后能真实跑通、不报错、产出符合预期。例如改了 FLUX 的 caching/forward 逻辑后,跑一张单图推理验证;改了 QWen 同理。
- **较简单的任务**:单图推理、改动后的快速 smoke test、参数验证等,直接自主完成运行验证,无需逐步询问用户。
- **复杂 / 高代价任务**:大批量推理、长耗时训练/评测、需要多卡、或会长时间占用大量显存(可能影响他人)的任务,先向用户说明计划与所需资源,得到确认后再执行,避免盲目占用。

## Reference Docs & Known Gotchas

- **[`docs/GOTCHAS.md`](docs/GOTCHAS.md)** — 可复用的坑 & 修复笔记。接入新模型 / 动 `run.py` 的 `--benchmark`、`--eval` 路径前先看,要点:
  1. raw-FLUX 的 `--no_cache` 是参数式缓存(没有开关)——真正的 baseline 要 `interval=1`(通过 `ModelRunner.no_cache_baseline` 注入),光跳过缓存旋钮不行;
  2. `run.py --eval` 默认 `XDG_CACHE_HOME=/data/public/.cache`,**本机 `/data` 不存在**——需先 `export HF_HOME` / `XDG_CACHE_HOME` 指向 `$HOME/.cache/...`,否则 ImageReward/BLIP 下载 `bert-base-uncased` 失败;
  3. 入口脚本的 `--benchmark` 早返回块里用到的每个变量都必须在块**之前**绑定(否则 `UnboundLocalError`;`flux/taylorseer/src/sample.py` 的 `prompts`/`base_seed` 曾中招,已修)。
- 其它文档:`docs/BENCHMARK_FIXES.md`(GEdit-Bench)、`docs/FLUX_SCHNELL_TAYLORSEER.md`、`docs/FLUX_LoRA.md`、`docs/FLUX_QUANT.md`、`docs/TaylorSeer_Smooth_Algorithm.md`、`docs/QWEN_TAYLORSEER_BENCHMARK.md`、`docs/EVAL_GEDIT_WORKFLOW.md`。
- 范例扫描脚本(DrawBench200 + eval + benchmark,扫 N/O/F/Alpha):`scripts/flux_schnell_sweep.sh`(`DRY_RUN=1` 可预览全部命令)。
