"""
Shared inference utilities for Cache4Diffusion.
Model-agnostic pipeline setup, usable by scripts and tests alike.
"""

import re
import torch
from diffusers import DiffusionPipeline
from model_adapters.factory import patch_model_with_cache

def sanitize_filename(text: str, max_length: int = 80) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("/", "-")
    text = re.sub(r"[^\w\-\s]", "", text)
    text = text.replace(" ", "_")
    if len(text) == 0:
        text = "prompt"
    return text[:max_length]

def get_torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def setup_pipeline(model_path: str, steps: int, strategy_name: str, model_name: str,
                   dtype: torch.dtype, device: str,
                   use_device_map: bool = False) -> DiffusionPipeline:
    """Load a diffusers pipeline and apply the caching framework.

    Args:
        model_path:      HuggingFace model id or local path.
        steps:           Number of inference steps (stored on transformer class).
        strategy_name:   Caching strategy, e.g. 'taylorseer'.
        model_name:      Adapter name, e.g. 'flux' or 'qwen_image'.
        dtype:           Torch dtype for model weights.
        device:          Target device string, e.g. 'cuda' or 'cpu'.
        use_device_map:  If True, load with device_map='cuda' instead of .to(device).
                         Required for models like QwenImage that use device_map loading.

    Returns:
        Pipeline ready for inference.
    """
    if use_device_map:
        pipeline = DiffusionPipeline.from_pretrained(
            model_path, torch_dtype=dtype, device_map="cuda"
        )
    else:
        pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipeline.transformer.__class__.num_steps = steps
    patch_model_with_cache(pipeline.transformer, strategy_name=strategy_name, model_name=model_name)
    if hasattr(pipeline, 'vae'):
        pipeline.vae.enable_tiling()
    if not use_device_map:
        pipeline.to(device)
    return pipeline
