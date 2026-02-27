"""
Model structure information.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelInfo:
    """Model structure info for strategy cache initialization."""
    num_double_layers: int = 0
    num_single_layers: int = 0
    has_double_stream: bool = True
    has_single_stream: bool = True
    num_steps: int = 50
    double_modules: Tuple[str, ...] = ('img_attn', 'img_mlp', 'txt_attn', 'txt_mlp')
    single_modules: Tuple[str, ...] = ('total',)
