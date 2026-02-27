"""
Base adapter interface for model architectures.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import torch

from .info import ModelInfo
from caching_core import CacheStrategy


class ModelAdapter(ABC):
    """Base class for model-specific adapters."""

    @abstractmethod
    def get_model_info(self, model) -> ModelInfo:
        """Extract model structure information."""
        ...

    @abstractmethod
    def get_block_iterators(self, model) -> Dict[str, List]:
        """
        Return {stream_name: [block_list]} mapping.
        e.g., FLUX: {'double_stream': model.transformer_blocks,
                     'single_stream': model.single_transformer_blocks}
        """
        ...

    @abstractmethod
    def forward_double_block_full(
        self, block, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor, rotary_emb, strategy: CacheStrategy,
        cache_dic: Dict, ctx: Any, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute double stream block full forward.
        Call strategy.on_full_compute() at each module output.
        Return (hidden_states, encoder_hidden_states).
        """
        ...

    @abstractmethod
    def forward_double_block_cached(
        self, block, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor, rotary_emb, strategy: CacheStrategy,
        cache_dic: Dict, ctx: Any, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute double stream block cache restore."""
        ...

    @abstractmethod
    def forward_single_block_full(
        self, block, hidden_states: torch.Tensor, temb: torch.Tensor,
        rotary_emb, strategy: CacheStrategy, cache_dic: Dict, ctx: Any, **kwargs
    ) -> torch.Tensor:
        """Execute single stream block full forward."""
        ...

    @abstractmethod
    def forward_single_block_cached(
        self, block, hidden_states: torch.Tensor, temb: torch.Tensor,
        rotary_emb, strategy: CacheStrategy, cache_dic: Dict, ctx: Any, **kwargs
    ) -> torch.Tensor:
        """Execute single stream block cache restore."""
        ...

    def create_forward_fn(self, model, strategy: CacheStrategy):
        """
        Generate monkey-patch forward function.
        Integration point with diffusers pipeline.
        """
        model_info = self.get_model_info(model)
        adapter = self

        def patched_forward(hidden_states, encoder_hidden_states=None,
                           temb=None, image_rotary_emb=None, joint_attention_kwargs=None, **kwargs):
            if joint_attention_kwargs is None:
                joint_attention_kwargs = {}

            # Initialize cache on model instance (persists across steps)
            if not hasattr(model, '_ts_cache_dic'):
                from taylorseer_core import cache_init
                cache_dic = cache_init(model_info.num_double_layers, model_info.num_single_layers)
                # Initialize step context
                from caching_core import StepContext
                ctx = StepContext(num_steps=model_info.num_steps)
                model._ts_cache_dic = cache_dic
                model._ts_ctx = ctx

            cache_dic = model._ts_cache_dic
            ctx = model._ts_ctx

            # Schedule
            strategy.schedule_step(cache_dic, ctx)

            # Iterate blocks
            block_iters = adapter.get_block_iterators(model)

            for stream_name, blocks in block_iters.items():
                ctx.stream = stream_name
                for idx, block in enumerate(blocks):
                    ctx.layer = idx
                    if stream_name == 'double_stream':
                        if ctx.type == 'full':
                            hidden_states, encoder_hidden_states = \
                                adapter.forward_double_block_full(
                                    block, hidden_states, encoder_hidden_states,
                                    temb, image_rotary_emb, strategy, cache_dic, ctx, **kwargs)
                        else:
                            hidden_states, encoder_hidden_states = \
                                adapter.forward_double_block_cached(
                                    block, hidden_states, encoder_hidden_states,
                                    temb, image_rotary_emb, strategy, cache_dic, ctx, **kwargs)
                    elif stream_name == 'single_stream':
                        if ctx.type == 'full':
                            hidden_states = \
                                adapter.forward_single_block_full(
                                    block, hidden_states, temb, image_rotary_emb,
                                    strategy, cache_dic, ctx, **kwargs)
                        else:
                            hidden_states = \
                                adapter.forward_single_block_cached(
                                    block, hidden_states, temb, image_rotary_emb,
                                    strategy, cache_dic, ctx, **kwargs)

                    # Hook after block
                    strategy.on_block_end(cache_dic, ctx, hidden_states)

            ctx.step += 1

            # Free cache after last denoising step to reclaim VRAM
            if ctx.step >= ctx.num_steps and hasattr(model, '_ts_cache_dic'):
                del model._ts_cache_dic, model._ts_ctx
                import torch
                torch.cuda.empty_cache()

            return hidden_states

        return patched_forward
