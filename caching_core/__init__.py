"""
Cache4Diffusion Strategy Layer

Defines the CacheStrategy interface for different caching methods.
"""

from .base import CacheStrategy
from .context import StepContext, CacheStore

__all__ = ['CacheStrategy', 'StepContext', 'CacheStore']
