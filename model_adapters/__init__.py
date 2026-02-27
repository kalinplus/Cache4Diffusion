"""
Cache4Diffusion Adapter Layer

Defines the ModelAdapter interface for different model architectures.
"""

from .base import ModelAdapter
from .info import ModelInfo

__all__ = ['ModelAdapter', 'ModelInfo']
