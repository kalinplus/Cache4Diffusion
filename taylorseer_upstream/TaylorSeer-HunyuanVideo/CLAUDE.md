# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TaylorSeer-HunyuanVideo is a fork of Tencent's HunyuanVideo, an open-source text-to-video generation model (13B+ parameters). It generates ~5-second videos (129 frames at 24fps) from text prompts using a diffusion transformer architecture with flow matching.

## Commands

### Install
```bash
conda create -n HunyuanVideo python==3.10.9 && conda activate HunyuanVideo
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
pip install ninja && pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3
pip install xfuser==0.4.0  # for multi-GPU parallel inference
```

### Download models
```bash
huggingface-cli download tencent/HunyuanVideo --local-dir ./ckpts
```

### Inference (single GPU)
```bash
python sample_video.py \
    --video-size 720 1280 --video-length 129 --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --flow-reverse --use-cpu-offload --save-path ./results
```

### Inference (multi-GPU via torchrun)
```bash
torchrun --nproc_per_node=8 sample_video.py \
    --video-size 1280 720 --video-length 129 --infer-steps 50 \
    --prompt "..." --flow-reverse --seed 42 \
    --ulysses-degree 8 --ring-degree 1 --save-path ./results
```

### Gradio web UI
```bash
python gradio_server.py --flow-reverse
SERVER_NAME=0.0.0.0 SERVER_PORT=8081 python gradio_server.py --flow-reverse
```

### Tests
```bash
pytest tests/test_attention.py
```

### VBench evaluation
```bash
bash scripts/run_sample_video_vbench.sh
```

## Architecture

### Core pipeline flow
```
Prompt → TextEncoder (MLLM + CLIP) → HunyuanVideoPipeline (diffusion transformer + flow matching scheduler) → 3D VAE decoder → Video
```

### `hyvideo/` package structure

- **`inference.py`** — `HunyuanVideoSampler` is the main entry point. Handles model loading, parallel inference setup (xDiT), and coordinates the full generation pipeline. Entry for both CLI (`sample_video.py`) and Gradio (`gradio_server.py`).
- **`config.py`** — All inference arguments (video size, steps, guidance, parallel settings). Parsed via `parse_args()`.
- **`constants.py`** — Model paths (configurable via `MODEL_BASE` env var, defaults to `./ckpts`), prompt templates, negative prompts, precision types.
- **`diffusion/pipelines/pipeline_hunyuan_video.py`** — Core `HunyuanVideoPipeline` extending diffusers' `DiffusionPipeline`. Orchestrates denoising loop with flow matching.
- **`diffusion/schedulers/scheduling_flow_match_discrete.py`** — Flow matching discrete scheduler for the denoising process.
- **`modules/models.py`** — `HYVideoDiffusionTransformer`: the 13B+ parameter diffusion transformer (dual-stream to single-stream architecture). Config lookup via `HUNYUAN_VIDEO_CONFIG`.
- **`modules/attention.py`** — Attention mechanisms including distributed variants for multi-GPU.
- **`modules/posemb_layers.py`** — 3D rotary positional embeddings (RoPE) for temporal/spatial dimensions.
- **`modules/token_refiner.py`** — Token refinement layer (LI-DiT style).
- **`modules/fp8_optimization.py`** — FP8 linear layer quantization (~10GB memory savings).
- **`modules/cache_functions/`** — Caching, attention optimization, token merging, and support set selection for efficient inference.
- **`modules/taylor_utils/`** — Taylor series approximation utilities (FLOPs conversion).
- **`text_encoder/__init__.py`** — Dual text encoding: MLLM (Llama-based `llava-llama-3-8b-v1_1`) for semantic understanding + CLIP (`clip-vit-large-patch14`) for visual features. Applies prompt templates from constants.
- **`vae/`** — 3D Causal VAE (4x temporal, 8x spatial, 16x channel compression). Encoder/decoder with tiling support for memory efficiency.
- **`prompt_rewrite.py`** — Prompt enhancement in Normal/Master modes.

### Model checkpoint layout (`./ckpts/`)
```
ckpts/
├── hunyuan-video-t2v-720p/   # Main diffusion model + VAE
│   └── vae/
├── text_encoder/              # MLLM (llava-llama-3-8b-v1_1-transformers)
└── text_encoder_2/            # CLIP (clip-vit-large-patch14)
```

### Key configuration arguments
- `--video-size H W` — Resolution (supported: 540p and 720p presets, various aspect ratios)
- `--video-length` — 65 or 129 frames
- `--infer-steps` — Denoising steps (default 50)
- `--flow-reverse` — Enable flow matching reverse process
- `--embedded-cfg-scale` — Embedded classifier-free guidance (default 6.0)
- `--flow-shift` — Flow shift factor (default 7.0)
- `--use-cpu-offload` — Trade speed for lower GPU memory
- `--ulysses-degree` / `--ring-degree` — Multi-GPU parallelism degrees (product = total GPUs)

### Hardware requirements
- Minimum 45GB GPU memory (540p), 60GB (720p)
- Recommended 80GB (tested on A100)
- NVIDIA GPU with CUDA 11.8 or 12.4

## Key dependencies
- PyTorch 2.4.0, diffusers 0.31.0, transformers 4.46.3
- Flash Attention v2.6.3 (required for attention)
- xfuser 0.4.0 (multi-GPU parallel inference)
- gradio 5.0.0 (web UI)

## Code Style

### Fail fast, no silent fallbacks
- In shell scripts: always use `set -euo pipefail`. Assign variables directly (e.g. `VAR="value"`), never use `${VAR:-default}` fallback syntax.
- In Python: prefer explicit configuration over implicit defaults. Configuration parameters passed via environment variables should be read without defaults where the intent is to catch missing config early.
- Rationale: silent defaults mask configuration errors. If a required parameter is missing, the script should fail immediately with a clear error rather than silently proceeding with a wrong value.
