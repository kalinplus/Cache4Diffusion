"""
Factory for creating strategy + adapter combinations.
"""

from typing import Any
from caching_core.base import CacheStrategy
from model_adapters.base import ModelAdapter


def create_caching_pipeline(
    model: Any,
    strategy_name: str = 'taylorseer',
    model_name: str = 'flux',
) -> Any:
    """
    Create a caching pipeline by wiring strategy + adapter.

    Args:
        model: The transformer model (e.g., FluxTransformer2DModel)
        strategy_name: 'taylorseer', 'clusca', 'speca'
        model_name: 'flux', 'qwen_image', 'hunyuan_video'

    Returns:
        A patched forward function ready for monkey-patching
    """
    # Import strategy
    if strategy_name == 'taylorseer':
        from caching_core.strategies import TaylorSeerStrategy
        from taylorseer_core.config import TaylorSeerConfig
        strategy = TaylorSeerStrategy(config=TaylorSeerConfig.from_env())
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    # Import adapter
    if model_name == 'flux':
        from model_adapters.adapters import FluxAdapter
        adapter = FluxAdapter()
    elif model_name == 'qwen_image':
        from model_adapters.adapters import QwenImageAdapter
        adapter = QwenImageAdapter()
    elif model_name == 'hunyuan_video':
        from model_adapters.adapters import HunyuanVideoAdapter
        adapter = HunyuanVideoAdapter()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Create patched forward function
    return adapter.create_forward_fn(model, strategy)


def patch_model_with_cache(
    model: Any,
    strategy_name: str = 'taylorseer',
    model_name: str = 'flux',
) -> None:
    """
    Directly monkey-patch a model with caching.

    Args:
        model: The transformer model to patch
        strategy_name: 'taylorseer', 'clusca', 'speca'
        model_name: 'flux', 'qwen_image', 'hunyuan_video'
    """
    forward_fn = create_caching_pipeline(model, strategy_name, model_name)

    # Patch the appropriate forward method
    model._forward = model.forward
    model.forward = forward_fn


def unpatch_model(model: Any, model_name: str = 'flux') -> None:
    """Restore original forward method."""
    if hasattr(model, '_forward'):
        model.forward = model._forward
        delattr(model, '_forward')
