"""
Base strategy interface for caching methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import torch

from .context import StepContext


class CacheStrategy(ABC):
    """Base class for caching method strategies."""

    @abstractmethod
    def schedule_step(self, cache_dic: Dict, ctx: StepContext) -> None:
        """
        Decide computation type for current step, set ctx.type.
        Corresponds to existing cal_type() function.
        """
        ...

    @abstractmethod
    def on_full_compute(
        self, cache_dic: Dict, ctx: StepContext, feature: torch.Tensor
    ) -> torch.Tensor:
        """
        On full step: store feature to cache, return original feature.
        Corresponds to derivative_approximation + smoothing logic.
        """
        ...

    @abstractmethod
    def on_cache_restore(
        self, cache_dic: Dict, ctx: StepContext
    ) -> torch.Tensor:
        """
        On cache step: restore approximate value from cache.
        Corresponds to taylor_formula.
        """
        ...

    def on_partial_update(
        self,
        cache_dic: Dict,
        ctx: StepContext,
        block: Any,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Optional: partial token refresh (ClusCa's cache_cutfresh + update_cache).
        Return None to indicate no partial update support.
        """
        return None

    def on_block_end(
        self, cache_dic: Dict, ctx: StepContext, hidden_states: torch.Tensor
    ) -> None:
        """
        Optional hook: callback after block ends (e.g., ClusCa's get_cluster_info).
        """
        pass
