"""
HunyuanImage 2.1 model adapter for diffusers HunyuanImageTransformer2DModel.

Block structure is nearly identical to FLUX:
- Double blocks (HunyuanImageTransformerBlock): AdaLayerNormZero + joint attention + FF
- Single blocks (HunyuanImageSingleTransformerBlock): AdaLayerNormZeroSingle + joint attention + proj_out

Key difference from FLUX: single blocks take (hidden_states, encoder_hidden_states)
separately, concat internally, and return both. No pre-concatenation between
double and single block loops.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import torch

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

from model_adapters.base import ModelAdapter
from model_adapters.info import ModelInfo
from caching_core import CacheStrategy


class HunyuanImageAdapter(ModelAdapter):
    """HunyuanImage 2.1 adapter for diffusers HunyuanImageTransformer2DModel."""

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
        """HunyuanImage-specific forward with full embedding preprocessing."""
        model_info = self.get_model_info(model)
        adapter = self

        def patched_forward(
            hidden_states: torch.Tensor,
            timestep: torch.LongTensor,
            encoder_hidden_states: torch.Tensor,
            encoder_attention_mask: torch.Tensor,
            timestep_r: Optional[torch.LongTensor] = None,
            encoder_hidden_states_2: Optional[torch.Tensor] = None,
            encoder_attention_mask_2: Optional[torch.Tensor] = None,
            guidance: Optional[torch.Tensor] = None,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            return_dict: bool = True,
        ) -> Union[torch.Tensor, Transformer2DModelOutput]:

            # Initialize cache
            if not hasattr(model, '_ts_cache_dic'):
                from taylorseer_core import cache_init
                from caching_core import StepContext
                model._ts_cache_dic = cache_init(
                    model_info.num_double_layers,
                    model_info.num_single_layers,
                )
                model._ts_ctx = StepContext(num_steps=model.num_steps)

            cache_dic = model._ts_cache_dic
            ctx = model._ts_ctx

            strategy.schedule_step(cache_dic, ctx)

            # LoRA scale handling
            if attention_kwargs is not None:
                attention_kwargs = attention_kwargs.copy()
                lora_scale = attention_kwargs.pop("scale", 1.0)
            else:
                lora_scale = 1.0

            if USE_PEFT_BACKEND:
                scale_lora_layers(model, lora_scale)

            # --- Embedding preprocessing (mirrors diffusers forward) ---
            if hidden_states.ndim == 4:
                batch_size, _channels, height, width = hidden_states.shape
                sizes = (height, width)
            elif hidden_states.ndim == 5:
                batch_size, _channels, frame, height, width = hidden_states.shape
                sizes = (frame, height, width)
            else:
                raise ValueError(
                    f"hidden_states must be 4D or 5D, got {hidden_states.shape}"
                )

            post_patch_sizes = tuple(
                d // p
                for d, p in zip(sizes, model.config.patch_size)
            )

            # 1. RoPE
            image_rotary_emb = model.rope(hidden_states)

            # 2. Conditional embeddings
            encoder_attention_mask = encoder_attention_mask.bool()
            temb = model.time_guidance_embed(
                timestep, guidance=guidance, timestep_r=timestep_r,
            )
            hidden_states = model.x_embedder(hidden_states)
            encoder_hidden_states = model.context_embedder(
                encoder_hidden_states, timestep, encoder_attention_mask,
            )

            # ByT5 second encoder support
            if (
                model.context_embedder_2 is not None
                and encoder_hidden_states_2 is not None
            ):
                encoder_hidden_states_2 = model.context_embedder_2(
                    encoder_hidden_states_2
                )
                encoder_attention_mask_2 = encoder_attention_mask_2.bool()

                new_enc_hs = []
                new_enc_mask = []
                for text, text_mask, text_2, text_mask_2 in zip(
                    encoder_hidden_states,
                    encoder_attention_mask,
                    encoder_hidden_states_2,
                    encoder_attention_mask_2,
                ):
                    new_enc_hs.append(torch.cat([
                        text_2[text_mask_2],
                        text[text_mask],
                        text_2[~text_mask_2],
                        text[~text_mask],
                    ], dim=0))
                    new_enc_mask.append(torch.cat([
                        text_mask_2[text_mask_2],
                        text_mask[text_mask],
                        text_mask_2[~text_mask_2],
                        text_mask[~text_mask],
                    ], dim=0))

                encoder_hidden_states = torch.stack(new_enc_hs)
                encoder_attention_mask = torch.stack(new_enc_mask)

            # Build attention mask
            attention_mask = torch.nn.functional.pad(
                encoder_attention_mask,
                (hidden_states.shape[1], 0),
                value=True,
            )
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # --- Double stream blocks ---
            ctx.stream = 'double_stream'
            for idx, block in enumerate(model.transformer_blocks):
                ctx.layer = idx
                if ctx.type == 'full':
                    hidden_states, encoder_hidden_states = (
                        adapter.forward_double_block_full(
                            block, hidden_states, encoder_hidden_states,
                            temb, image_rotary_emb, strategy, cache_dic, ctx,
                            attention_mask=attention_mask,
                        )
                    )
                else:
                    hidden_states, encoder_hidden_states = (
                        adapter.forward_double_block_cached(
                            block, hidden_states, encoder_hidden_states,
                            temb, image_rotary_emb, strategy, cache_dic, ctx,
                        )
                    )
                strategy.on_block_end(cache_dic, ctx, hidden_states)

            # --- Single stream blocks ---
            # HunyuanImage single blocks take (img, txt) separately,
            # concat internally, and return (img, txt).
            ctx.stream = 'single_stream'
            for idx, block in enumerate(model.single_transformer_blocks):
                ctx.layer = idx
                if ctx.type == 'full':
                    hidden_states, encoder_hidden_states = (
                        adapter.forward_single_block_full(
                            block, hidden_states, temb, image_rotary_emb,
                            strategy, cache_dic, ctx,
                            encoder_hidden_states=encoder_hidden_states,
                            attention_mask=attention_mask,
                        )
                    )
                else:
                    hidden_states, encoder_hidden_states = (
                        adapter.forward_single_block_cached(
                            block, hidden_states, temb, image_rotary_emb,
                            strategy, cache_dic, ctx,
                            encoder_hidden_states=encoder_hidden_states,
                        )
                    )
                strategy.on_block_end(cache_dic, ctx, hidden_states)

            # --- Output projection ---
            hidden_states = model.norm_out(hidden_states, temb)
            hidden_states = model.proj_out(hidden_states)

            # --- Unpatchify ---
            out_channels = model.config.out_channels
            reshape_dims = (
                [batch_size]
                + list(post_patch_sizes)
                + [out_channels]
                + list(model.config.patch_size)
            )
            hidden_states = hidden_states.reshape(*reshape_dims)

            ndim = len(post_patch_sizes)
            permute_pattern = [0, ndim + 1]
            for i in range(ndim):
                permute_pattern.extend([i + 1, ndim + 2 + i])
            hidden_states = hidden_states.permute(*permute_pattern)

            final_dims = [batch_size, out_channels] + [
                pp * ps
                for pp, ps in zip(post_patch_sizes, model.config.patch_size)
            ]
            hidden_states = hidden_states.reshape(*final_dims)

            if USE_PEFT_BACKEND:
                unscale_lora_layers(model, lora_scale)

            ctx.step += 1

            # Free cache after last denoising step
            if ctx.step >= ctx.num_steps and hasattr(model, '_ts_cache_dic'):
                del model._ts_cache_dic, model._ts_ctx
                torch.cuda.empty_cache()

            if not return_dict:
                return (hidden_states,)
            return Transformer2DModelOutput(sample=hidden_states)

        return patched_forward

    # ------------------------------------------------------------------
    # Double stream block (identical pattern to FLUX)
    # ------------------------------------------------------------------

    def forward_double_block_full(
        self, block, hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor, rotary_emb, strategy: CacheStrategy,
        cache_dic: Dict, ctx: Any, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward for HunyuanImageTransformerBlock."""
        attention_mask = kwargs.get('attention_mask')

        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            block.norm1(hidden_states, emb=temb)
        )
        norm_enc_hs, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            block.norm1_context(encoder_hidden_states, emb=temb)
        )

        # Joint attention
        attn_output, context_attn_output = block.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_enc_hs,
            attention_mask=attention_mask,
            image_rotary_emb=rotary_emb,
        )

        # Image stream
        ctx.module = 'img_attn'
        strategy.on_full_compute(cache_dic, ctx, attn_output)
        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)

        ctx.module = 'img_mlp'
        norm_hidden_states = block.norm2(hidden_states)
        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )
        ff_output = block.ff(norm_hidden_states)
        strategy.on_full_compute(cache_dic, ctx, ff_output)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        # Text stream
        ctx.module = 'txt_attn'
        strategy.on_full_compute(cache_dic, ctx, context_attn_output)
        encoder_hidden_states = (
            encoder_hidden_states
            + context_attn_output * c_gate_msa.unsqueeze(1)
        )

        ctx.module = 'txt_mlp'
        norm_enc_hs = block.norm2_context(encoder_hidden_states)
        norm_enc_hs = (
            norm_enc_hs * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        )
        context_ff_output = block.ff_context(norm_enc_hs)
        strategy.on_full_compute(cache_dic, ctx, context_ff_output)
        encoder_hidden_states = (
            encoder_hidden_states
            + c_gate_mlp.unsqueeze(1) * context_ff_output
        )

        return hidden_states, encoder_hidden_states

    def forward_double_block_cached(
        self, block, hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor, rotary_emb, strategy: CacheStrategy,
        cache_dic: Dict, ctx: Any, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cache restore for HunyuanImageTransformerBlock."""
        del rotary_emb, kwargs  # unused in cached path
        _, gate_msa, _, _, gate_mlp = (
            block.norm1(hidden_states, emb=temb)
        )
        _, c_gate_msa, _, _, c_gate_mlp = (
            block.norm1_context(encoder_hidden_states, emb=temb)
        )

        ctx.module = 'img_attn'
        hidden_states = (
            hidden_states
            + strategy.on_cache_restore(cache_dic, ctx) * gate_msa.unsqueeze(1)
        )

        ctx.module = 'img_mlp'
        hidden_states = (
            hidden_states
            + gate_mlp.unsqueeze(1) * strategy.on_cache_restore(cache_dic, ctx)
        )

        ctx.module = 'txt_attn'
        encoder_hidden_states = (
            encoder_hidden_states
            + strategy.on_cache_restore(cache_dic, ctx) * c_gate_msa.unsqueeze(1)
        )

        ctx.module = 'txt_mlp'
        encoder_hidden_states = (
            encoder_hidden_states
            + c_gate_mlp.unsqueeze(1) * strategy.on_cache_restore(cache_dic, ctx)
        )

        return hidden_states, encoder_hidden_states

    # ------------------------------------------------------------------
    # Single stream block
    # ------------------------------------------------------------------

    def forward_single_block_full(
        self, block, hidden_states: torch.Tensor, temb: torch.Tensor,
        rotary_emb, strategy: CacheStrategy, cache_dic: Dict, ctx: Any,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward for HunyuanImageSingleTransformerBlock."""
        encoder_hidden_states = kwargs['encoder_hidden_states']
        attention_mask = kwargs.get('attention_mask')
        text_seq_length = encoder_hidden_states.shape[1]

        # Concat img + txt
        hidden_states = torch.cat(
            [hidden_states, encoder_hidden_states], dim=1,
        )
        residual = hidden_states

        # Norm + gate
        norm_hidden_states, gate = block.norm(hidden_states, emb=temb)
        mlp_hidden_states = block.act_mlp(block.proj_mlp(norm_hidden_states))

        # Split for attention (RoPE only on image tokens)
        norm_hs, norm_enc = (
            norm_hidden_states[:, :-text_seq_length, :],
            norm_hidden_states[:, -text_seq_length:, :],
        )

        # Joint attention
        attn_output, context_attn_output = block.attn(
            hidden_states=norm_hs,
            encoder_hidden_states=norm_enc,
            attention_mask=attention_mask,
            image_rotary_emb=rotary_emb,
        )
        attn_output = torch.cat([attn_output, context_attn_output], dim=1)

        # Project
        proj_input = torch.cat([attn_output, mlp_hidden_states], dim=2)
        proj_output = block.proj_out(proj_input)

        ctx.module = 'total'
        strategy.on_full_compute(cache_dic, ctx, proj_output)

        hidden_states = gate.unsqueeze(1) * proj_output + residual

        # Split back
        hidden_states, encoder_hidden_states = (
            hidden_states[:, :-text_seq_length, :],
            hidden_states[:, -text_seq_length:, :],
        )
        return hidden_states, encoder_hidden_states

    def forward_single_block_cached(
        self, block, hidden_states: torch.Tensor, temb: torch.Tensor,
        rotary_emb, strategy: CacheStrategy, cache_dic: Dict, ctx: Any,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cache restore for HunyuanImageSingleTransformerBlock."""
        del rotary_emb  # unused in cached path
        encoder_hidden_states = kwargs['encoder_hidden_states']
        text_seq_length = encoder_hidden_states.shape[1]

        # Concat img + txt
        hidden_states = torch.cat(
            [hidden_states, encoder_hidden_states], dim=1,
        )
        residual = hidden_states

        # Gate from norm
        _, gate = block.norm(hidden_states, emb=temb)

        ctx.module = 'total'
        proj_output = strategy.on_cache_restore(cache_dic, ctx)

        hidden_states = gate.unsqueeze(1) * proj_output + residual

        # Split back
        hidden_states, encoder_hidden_states = (
            hidden_states[:, :-text_seq_length, :],
            hidden_states[:, -text_seq_length:, :],
        )
        return hidden_states, encoder_hidden_states
