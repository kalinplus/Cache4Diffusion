# Thin wrapper - re-exports from shared taylorseer_core for backward compatibility
from taylorseer_core.math import (
    derivative_approximation,
    derivative_approximation_with_smoothing,
    derivative_approximation_hybrid_smoothing,
    taylor_formula,
    module_cache_init,
    module_cache_init as taylor_cache_init,
    exponential_smoothing,
    moving_average_smoothing,
    shift_cache_history,
)

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
]
