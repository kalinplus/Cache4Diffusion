"""
FLUX model adapter implementation.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, scale_lora_layers, unscale_lora_layers

from model_adapters.base import ModelAdapter
from model_adapters.info import ModelInfo
from caching_core import CacheStrategy


class FluxAdapter(ModelAdapter):
    """FLUX model adapter."""

    def get_model_info(self, model) -> ModelInfo:
        return ModelInfo(
            num_double_layers=model.config.num_layers,
            num_single_layers=model.config.num_single_layers,
            has_double_stream=True,
            has_single_stream=True,
        )

    def get_block_iterators(self, model) -> Dict[str, List]:
        return {
            'double_stream': list(model.transformer_blocks),
            'single_stream': list(model.single_transformer_blocks),
        }

    def create_forward_fn(self, model, strategy: CacheStrategy):
        """FLUX-specific forward: includes full embedding preprocessing."""
        from diffusers.utils import logging as diffusers_logging
        logger = diffusers_logging.get_logger(__name__)

        model_info = self.get_model_info(model)
        adapter = self

        def patched_forward(
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            pooled_projections: torch.Tensor = None,
            timestep: torch.LongTensor = None,
            img_ids: torch.Tensor = None,
            txt_ids: torch.Tensor = None,
            guidance: torch.Tensor = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            controlnet_block_samples=None,
            controlnet_single_block_samples=None,
            return_dict: bool = True,
            controlnet_blocks_repeat: bool = False,
        ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:

            if joint_attention_kwargs is None:
                joint_attention_kwargs = {}

            # Initialize cache on model instance (persists across steps)
            if not hasattr(model, '_ts_cache_dic'):
                from taylorseer_core import cache_init
                from caching_core import StepContext
                model._ts_cache_dic = cache_init(model_info.num_double_layers, model_info.num_single_layers)
                model._ts_ctx = StepContext(num_steps=model.num_steps)

            cache_dic = model._ts_cache_dic
            ctx = model._ts_ctx

            # Schedule step type
            strategy.schedule_step(cache_dic, ctx)

            # LoRA scale handling
            if joint_attention_kwargs is not None:
                joint_attention_kwargs = joint_attention_kwargs.copy()
                lora_scale = joint_attention_kwargs.pop("scale", 1.0)
            else:
                lora_scale = 1.0

            if USE_PEFT_BACKEND:
                scale_lora_layers(model, lora_scale)

            # FLUX-specific embeddings
            hidden_states = model.x_embedder(hidden_states)

            timestep = timestep.to(hidden_states.dtype) * 1000
            if guidance is not None:
                guidance = guidance.to(hidden_states.dtype) * 1000

            temb = (
                model.time_text_embed(timestep, pooled_projections)
                if guidance is None
                else model.time_text_embed(timestep, guidance, pooled_projections)
            )
            encoder_hidden_states = model.context_embedder(encoder_hidden_states)

            if txt_ids.ndim == 3:
                txt_ids = txt_ids[0]
            if img_ids.ndim == 3:
                img_ids = img_ids[0]

            ids = torch.cat((txt_ids, img_ids), dim=0)
            image_rotary_emb = model.pos_embed(ids)

            # Double stream blocks
            ctx.stream = 'double_stream'
            for idx, block in enumerate(model.transformer_blocks):
                ctx.layer = idx
                if ctx.type == 'full':
                    encoder_hidden_states, hidden_states = adapter.forward_double_block_full(
                        block, hidden_states, encoder_hidden_states,
                        temb, image_rotary_emb, strategy, cache_dic, ctx)
                else:
                    encoder_hidden_states, hidden_states = adapter.forward_double_block_cached(
                        block, hidden_states, encoder_hidden_states,
                        temb, image_rotary_emb, strategy, cache_dic, ctx)

                strategy.on_block_end(cache_dic, ctx, hidden_states)

                if controlnet_block_samples is not None:
                    interval = int(np.ceil(len(model.transformer_blocks) / len(controlnet_block_samples)))
                    if controlnet_blocks_repeat:
                        hidden_states = hidden_states + controlnet_block_samples[idx % len(controlnet_block_samples)]
                    else:
                        hidden_states = hidden_states + controlnet_block_samples[idx // interval]

            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

            # Single stream blocks
            ctx.stream = 'single_stream'
            for idx, block in enumerate(model.single_transformer_blocks):
                ctx.layer = idx
                if ctx.type == 'full':
                    hidden_states = adapter.forward_single_block_full(
                        block, hidden_states, temb, image_rotary_emb, strategy, cache_dic, ctx)
                else:
                    hidden_states = adapter.forward_single_block_cached(
                        block, hidden_states, temb, image_rotary_emb, strategy, cache_dic, ctx)

                strategy.on_block_end(cache_dic, ctx, hidden_states)

                if controlnet_single_block_samples is not None:
                    interval = int(np.ceil(len(model.single_transformer_blocks) / len(controlnet_single_block_samples)))
                    hidden_states[:, encoder_hidden_states.shape[1]:, ...] = (
                        hidden_states[:, encoder_hidden_states.shape[1]:, ...]
                        + controlnet_single_block_samples[idx // interval]
                    )

            hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]
            hidden_states = model.norm_out(hidden_states, temb)
            output = model.proj_out(hidden_states)

            if USE_PEFT_BACKEND:
                unscale_lora_layers(model, lora_scale)

            ctx.step += 1

            # Free cache after last denoising step to reclaim VRAM for VAE decode
            if ctx.step >= ctx.num_steps and hasattr(model, '_ts_cache_dic'):
                del model._ts_cache_dic, model._ts_ctx
                torch.cuda.empty_cache()

            if not return_dict:
                return (output,)
            return Transformer2DModelOutput(sample=output)

        return patched_forward

    def forward_double_block_full(
        self, block, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor, rotary_emb, strategy: CacheStrategy,
        cache_dic: Dict, ctx: Any, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward for double transformer block."""
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(
            hidden_states, emb=temb
        )
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = \
            block.norm1_context(encoder_hidden_states, emb=temb)

        attention_outputs = block.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=rotary_emb,
        )
        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        else:
            attn_output, context_attn_output = attention_outputs[0], attention_outputs[1]

        ctx.module = 'img_attn'
        strategy.on_full_compute(cache_dic, ctx, attn_output)
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        ctx.module = 'img_mlp'
        norm_hidden_states = block.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = block.ff(norm_hidden_states)
        strategy.on_full_compute(cache_dic, ctx, ff_output)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        ctx.module = 'txt_attn'
        strategy.on_full_compute(cache_dic, ctx, context_attn_output)
        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output

        ctx.module = 'txt_mlp'
        norm_encoder_hidden_states = block.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        context_ff_output = block.ff_context(norm_encoder_hidden_states)
        strategy.on_full_compute(cache_dic, ctx, context_ff_output)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    def forward_double_block_cached(
        self, block, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor, rotary_emb, strategy: CacheStrategy,
        cache_dic: Dict, ctx: Any, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cache restore for double transformer block."""
        _, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(hidden_states, emb=temb)
        _, c_gate_msa, _, _, c_gate_mlp = block.norm1_context(encoder_hidden_states, emb=temb)

        ctx.module = 'img_attn'
        attn_output = strategy.on_cache_restore(cache_dic, ctx)
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        ctx.module = 'img_mlp'
        ff_output = strategy.on_cache_restore(cache_dic, ctx)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        ctx.module = 'txt_attn'
        context_attn_output = strategy.on_cache_restore(cache_dic, ctx)
        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output

        ctx.module = 'txt_mlp'
        context_ff_output = strategy.on_cache_restore(cache_dic, ctx)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    def forward_single_block_full(
        self, block, hidden_states: torch.Tensor, temb: torch.Tensor,
        rotary_emb, strategy: CacheStrategy, cache_dic: Dict, ctx: Any, **kwargs
    ) -> torch.Tensor:
        """Full forward for single transformer block."""
        norm_hidden_states, gate = block.norm(hidden_states, emb=temb)
        gate = gate.unsqueeze(1)
        residual = hidden_states

        ctx.module = 'total'
        mlp_hidden_states = block.act_mlp(block.proj_mlp(norm_hidden_states))
        attn_output = block.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=rotary_emb,
        )
        proj_input = torch.cat([attn_output, mlp_hidden_states], dim=2)
        proj_output = block.proj_out(proj_input)
        strategy.on_full_compute(cache_dic, ctx, proj_output)

        hidden_states = residual + gate * proj_output

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states

    def forward_single_block_cached(
        self, block, hidden_states: torch.Tensor, temb: torch.Tensor,
        rotary_emb, strategy: CacheStrategy, cache_dic: Dict, ctx: Any, **kwargs
    ) -> torch.Tensor:
        """Cache restore for single transformer block."""
        _, gate = block.norm(hidden_states, emb=temb)
        gate = gate.unsqueeze(1)
        residual = hidden_states

        ctx.module = 'total'
        proj_output = strategy.on_cache_restore(cache_dic, ctx)

        hidden_states = residual + gate * proj_output

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states
