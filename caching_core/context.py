"""
Data structures for caching framework.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class StepContext:
    """Shared context per timestep (replaces 'current' dict).

    Supports dict-style access for backward compatibility with taylorseer_core
    functions that use current['key'] syntax.
    """
    step: int = 0
    num_steps: int = 50
    type: str = 'full'           # Set by strategy's schedule_step
    activated_steps: List[int] = field(default_factory=lambda: [0])
    stream: str = ''             # Set by adapter: 'double_stream' / 'single_stream'
    layer: int = 0               # Set by adapter
    module: str = ''             # Set by adapter: 'img_attn' / 'img_mlp' / ...

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value):
        setattr(self, key, value)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


@dataclass
class CacheStore:
    """Unified cache storage (replaces cache_dic['cache'])."""
    data: Dict  # cache[-1][stream][layer][module] -> {0: tensor, 1: tensor, ...}
    history: Optional[Dict] = None  # cache[-2], for smoothing only
    index: Optional[Dict] = None    # cache_index, for ToCa/ClusCa only
    extra: Dict = field(default_factory=dict)  # attn_map, k-norm, etc.
