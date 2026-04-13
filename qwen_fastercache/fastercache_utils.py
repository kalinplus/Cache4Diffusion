"""
FasterCache utilities for Qwen dual-stream DiT.

Usage
-----
    from fastercache_utils import pipeline_with_fastercache

    pipe = QwenImagePipeline.from_pretrained(...)
    pipe = pipeline_with_fastercache(pipe)          # use defaults
    # or
    pipe = pipeline_with_fastercache(
        pipe,
        start_step=15,     # steps 0..start_step always run full attention
        cache_interval=2,  # every 2nd step is a "full" step; others are skipped
        alpha=0.3,         # linear extrapolation coefficient (0 = plain reuse)
    )

Design
------
FasterCache replaces the expensive O(n^2) joint attention inside each
QwenImageTransformerBlock with a cheap cache lookup on most denoising steps.

Two forward passes happen per timestep in CFG mode ('cond' and 'uncond').
To keep their attention states independent we key the per-block cache by
the module string.

The patching strategy follows freqca's `pipeline_with_taylorseer`:
  - `pipe.transformer.forward`  → replaced with the FasterCache model forward
  - each block's `.forward`     → replaced with the FasterCache block forward
Both replacements use `types.MethodType` so `self` still refers to the
original instance, preserving all weights and config.
"""

import types
from typing import Optional


def pipeline_with_fastercache(
    pipe,
    start_step: int = 15,
    cache_interval: int = 2,
    alpha: float = 0.3,
):
    """
    Monkey-patch *pipe.transformer* and every *QwenImageTransformerBlock*
    inside it to enable FasterCache attention skipping.

    Parameters
    ----------
    pipe : QwenImagePipeline
        A loaded Qwen diffusion pipeline (weights must already be on device).
    start_step : int
        Cache warm-up length.  Steps with index 0 … start_step run full
        attention unconditionally so the cache is populated before skipping begins.
        Recommended: 15 for a 50-step schedule; scale linearly for other schedules.
    cache_interval : int
        Attention is recomputed every `cache_interval` steps (steps 0, interval,
        2*interval, …).  All other steps reuse/extrapolate cached values.
        cache_interval=2  → ~50 % attention skipped after warm-up  (Vchitect style)
        cache_interval=3  → ~67 % attention skipped after warm-up  (CogVideoX style)
    alpha : float
        Linear extrapolation coefficient applied on skip steps:
            attn ≈ cached_new + alpha * (cached_new - cached_old)
        0.0 → plain cache reuse (fastest, least accurate)
        0.3 → mild extrapolation (default, good quality/speed trade-off)

    Returns
    -------
    pipe : QwenImagePipeline
        The same pipe object with transformer forward methods patched in-place.
    """
    from pipeline.transformer_qwenimage import (
        QwenImageTransformer2DModel as FasterCacheModel,
        QwenImageTransformerBlock   as FasterCacheBlock,
    )

    transformer = pipe.transformer

    # 1. Patch each block's forward (instance-level to avoid global class mutation).
    for block in transformer.transformer_blocks:
        block.forward = types.MethodType(FasterCacheBlock.forward, block)

    # 2. Patch the model's forward.
    transformer.forward = types.MethodType(FasterCacheModel.forward, transformer)

    # 3. Attach reset_fastercache so the pipeline can clear caches between images.
    transformer.reset_fastercache = types.MethodType(FasterCacheModel.reset_fastercache, transformer)

    # 4. Store default config on the transformer so the pipeline can read it.
    transformer._fc_default_start_step    = start_step
    transformer._fc_default_cache_interval = cache_interval
    transformer._fc_default_alpha          = alpha

    return pipe
