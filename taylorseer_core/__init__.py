"""
taylorseer_core: Shared mathematical utilities for TaylorSeer caching.
"""
from .math import (
    derivative_approximation,
    derivative_approximation_with_smoothing,
    derivative_approximation_hybrid_smoothing,
    taylor_formula,
    module_cache_init,
    taylor_cache_init,
    exponential_smoothing,
    moving_average_smoothing,
    shift_cache_history,
    cache_init,
)
from .scheduler import cal_type, force_scheduler
from .config import TaylorSeerConfig
from .forward_utils import update_cache_or_approximate

__all__ = [
    "derivative_approximation",
    "derivative_approximation_with_smoothing",
    "derivative_approximation_hybrid_smoothing",
    "taylor_formula",
    "module_cache_init",
    "taylor_cache_init",
    "exponential_smoothing",
    "moving_average_smoothing",
    "shift_cache_history",
    "cache_init",
    "cal_type",
    "force_scheduler",
    "TaylorSeerConfig",
]
