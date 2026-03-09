# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a research repository for **inference acceleration of Qwen-Image diffusion models** via KV/activation caching. It contains multiple cache-method variants that can be benchmarked against a baseline.

## Environment Setup

```bash
conda activate qwen   # Python 3.10 environment
```

Key dependencies: `torch==2.6.0` (CUDA 12.4), `transformers==4.55.4`, `diffusers` (from git HEAD), `peft`.
Set `export HF_ENDPOINT="https://hf-mirror.com"` to use the HuggingFace mirror.

## Running Inference

All scripts are run from inside a variant directory (e.g., `cd taylorseer`), since they use relative imports from `pipeline/` and `cache_functions/`.

```bash
# Single GPU – text-to-image or image editing
CUDA_VISIBLE_DEVICES=0 python sample.py \
    --model_name qwen-image \          # or qwen-image-edit / qwen-image-lightning
    --prompt_file prompts/DrawBench200.txt \
    --output_dir samples/test \
    --interval 6 --max_order 2 --first_enhance 3

# Multi-GPU distributed (uses torchrun + NCCL)
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 sample_ddp.py \
    --model_name qwen-image --output_dir samples/test

# GEdit-Bench dataset generation (image editing, distributed)
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 sample_gedit.py \
    --dataset_path /path/to/GEdit-Bench --output_dir samples/gedit
```

## Evaluation

```bash
# Text-to-image quality (CLIP score, ImageReward, PSNR, SSIM, LPIPS)
python evaluate.py \
    --test_folder samples/test \
    --prompt_file prompts/DrawBench200.txt \
    --reference_folder samples/origin   # optional, for PSNR/SSIM/LPIPS

# GEdit-Bench evaluation
python evaluate_gedit.py --generated_dir samples/gedit
```

Evaluation requires models downloaded: `zai-org/ImageReward` and `laion/CLIP-ViT-g-14-laion2B-s12B-b42K`.

## Repository Structure

The repo has **one directory per caching method** plus a baseline:

| Directory | Cache Strategy |
|-----------|---------------|
| `qwen/` | Baseline – no caching, original pipeline |
| `taylorseer/` | Taylor expansion: stores activations + finite-difference derivatives, approximates skipped steps |
| `toca/` | Token caching: attention-map-scored token selection; only recomputes "fresh" tokens per step |
| `duca/` | Dual caching: token-level selection similar to ToCa with different scoring |
| `freqca/` | Frequency caching: FFT/DCT decomposition; caches low- and high-frequency components separately (`torch_dct` required) |
| `teacache/` | Threshold caching: residual-similarity gating baked into the transformer; no `cache_functions/` subpackage |

Each variant has the **same file layout**:
- `sample.py` / `sample_ddp.py` / `sample_gedit.py` – generation scripts
- `evaluate.py` / `evaluate_gedit.py` – evaluation scripts
- `pipeline/pipeline_qwenimage.py` – text-to-image pipeline (wraps diffusers)
- `pipeline/pipeline_qwenimage_edit.py` – image-editing pipeline
- `pipeline/transformer_qwenimage.py` – modified `QwenImageTransformer2DModel` with cache hooks
- `cache_functions/` – cache logic (absent in `teacache/` and `qwen/`)
- `viescore/` – VLM-based scoring using Gemini, OpenAI, or Qwen2.5-VL
- `prompts/` – prompt lists: `DrawBench200.txt`, `parti_prompts.txt`, `prompts_for_edit.txt`

## Cache Architecture

Cache injection is done via **monkey-patching** at runtime (`pipeline_with_cache(pipe)` in `cache_functions/cache_utils.py`). It replaces:
- `pipe.transformer.forward` with a local version that threads `cache_dic` / `current` dicts
- Each `QwenImageTransformerBlock.forward` with a cache-aware version

The `current` dict tracks `step`, `layer`, `module`, `stream` (cond/uncond), `type` (`full` or `cache`), and `activated_steps`. `cache_dic` holds cached tensors keyed by `cache[-1][stream][layer_idx][module]`.

The `cal_type()` function in `cache_functions/cal_type.py` determines for each denoising step whether to run full computation or read from cache, based on `interval` and `first_enhance`.

The transformer is a 60-layer dual-stream DiT (`QwenImageTransformerBlock`): image and text tokens are processed jointly in each block via `QwenDoubleStreamAttnProcessor2_0`, with separate MLP branches and RoPE positional embeddings (`QwenEmbedRope`).

## Supported Models

- `Qwen/Qwen-Image` – text-to-image
- `Qwen/Qwen-Image-Edit` – instruction-guided image editing
- `lightx2v/Qwen-Image-Lightning` – 8-step distilled model (LoRA on top of `Qwen-Image`; requires `--num_steps 8`)
