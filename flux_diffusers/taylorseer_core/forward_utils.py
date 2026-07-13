"""
Unified cache update utility for TaylorSeer forward passes.
Replaces the repeated if/elif/else blocks in each model's forward files.
"""
import torch
from typing import Dict, Optional

from taylorseer_core.math import (
    derivative_approximation,
    derivative_approximation_with_smoothing,
    derivative_approximation_hybrid_smoothing,
    taylor_formula,
    module_cache_init,
)


def update_cache_or_approximate(
    cache_dic: Dict,
    current: Dict,
    feature: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Unified cache update / Taylor approximation.

    In 'full' mode:
      - shift history (if smoothing enabled)
      - init cache slot
      - compute derivative approximation (with or without smoothing)
      - returns the original feature unchanged

    In 'Taylor' mode:
      - returns the Taylor approximation (feature argument is ignored)
    """
    if current['type'] == 'full':
        use_smoothing = cache_dic.get('use_smoothing', False)
        use_hybrid = cache_dic.get('use_hybrid_smoothing', False)
        method = cache_dic.get('smoothing_method', 'exponential')
        alpha = cache_dic.get('smoothing_alpha', 0.8)

        module_cache_init(cache_dic, current)

        if use_smoothing and use_hybrid:
            derivative_approximation_hybrid_smoothing(
                cache_dic, current, feature, method, alpha
            )
        elif use_smoothing:
            derivative_approximation_with_smoothing(
                cache_dic, current, feature, method, alpha
            )
        else:
            derivative_approximation(cache_dic, current, feature)

        return feature
    else:
        return taylor_formula(cache_dic, current)
