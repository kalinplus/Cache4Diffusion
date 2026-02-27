"""
TaylorSeer strategy implementation wrapping taylorseer_core.
"""

from typing import Dict
import torch

from caching_core.base import CacheStrategy
from caching_core.context import StepContext
from taylorseer_core.scheduler import cal_type
from taylorseer_core.forward_utils import update_cache_or_approximate, taylor_formula


class TaylorSeerStrategy(CacheStrategy):
    """TaylorSeer: fixed interval + Taylor series approximation + optional smoothing."""

    def __init__(self, config=None):
        """
        Args:
            config: TaylorSeerConfig or None (will use from_env())
        """
        if config is None:
            from taylorseer_core.config import TaylorSeerConfig
            config = TaylorSeerConfig.from_env()
        self.config = config

    def schedule_step(self, cache_dic: Dict, ctx: StepContext) -> None:
        """Decide step type using cal_type from taylorseer_core."""
        # cal_type modifies current dict in place, sets current['type']
        cal_type(cache_dic, ctx)

    def on_full_compute(
        self, cache_dic: Dict, ctx: StepContext, feature: torch.Tensor
    ) -> torch.Tensor:
        """Store feature to cache with optional smoothing."""
        return update_cache_or_approximate(cache_dic, ctx, feature)

    def on_cache_restore(self, cache_dic: Dict, ctx: StepContext) -> torch.Tensor:
        """Restore approximate value using Taylor formula."""
        return taylor_formula(cache_dic, ctx)
