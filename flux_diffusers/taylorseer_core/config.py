"""
Unified TaylorSeer configuration.
Fixes SMOOTHING_ALPHA default drift (Flux=0.7, QwenImage=0.8) -> unified 0.8.
"""
from dataclasses import dataclass
import os


@dataclass
class TaylorSeerConfig:
    cache_interval: int = 6
    max_order: int = 1
    first_enhance: int = 3
    use_smoothing: bool = False
    use_hybrid_smoothing: bool = False
    smoothing_method: str = "exponential"
    smoothing_alpha: float = 0.8  # unified default (fixes Flux 0.7 drift)

    @classmethod
    def from_env(cls) -> "TaylorSeerConfig":
        return cls(
            cache_interval=int(os.environ.get("TS_CACHE_INTERVAL", 6)),
            max_order=int(os.environ.get("TS_MAX_ORDER", 1)),
            first_enhance=int(os.environ.get("TS_FIRST_ENHANCE", 3)),
            use_smoothing=os.environ.get("USE_SMOOTHING", "False").lower() in ("true", "1", "yes"),
            use_hybrid_smoothing=os.environ.get("USE_HYBRID_SMOOTHING", "False").lower() == "true",
            smoothing_method=os.environ.get("SMOOTHING_METHOD", "exponential"),
            smoothing_alpha=float(os.environ.get("SMOOTHING_ALPHA", "0.8")),
        )
