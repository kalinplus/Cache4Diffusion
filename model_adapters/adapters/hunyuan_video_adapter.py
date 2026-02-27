"""
HunyuanVideo model adapter implementation.
Has both double-stream and single-stream transformer blocks.
Single-stream blocks concatenate img+txt, process jointly, then split.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import torch

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

from model_adapters.base import ModelAdapter
from model_adapters.info import ModelInfo
from caching_core import CacheStrategy

_DEBUG = os.environ.get("TS_DEBUG_FORWARD", "0").lower() in ("1", "true", "yes")


def _gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return 0.0


def _dbg(msg):
    if _DEBUG:
        print(f"[HYV-DBG] {msg}", flush=True)


class HunyuanVideoAdapter(ModelAdapter):
    """HunyuanVideo model adapter. Double-stream + single-stream blocks."""

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
        """HunyuanVideo-specific forward: video preprocessing, attention mask, reshape."""
        model_info = self.get_model_info(model)
        adapter = self

        def patched_forward(
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            encoder_attention_mask: torch.Tensor = None,
            pooled_projections: torch.Tensor = None,
            timestep: torch.LongTensor = None,
            guidance: torch.Tensor = None,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            return_dict: bool = True,
        ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:

            # Initialize cache on model instance (persists across steps)
            if not hasattr(model, '_ts_cache_dic'):
                from taylorseer_core import cache_init
                from caching_core import StepContext
                model._ts_cache_dic = cache_init(model_info.num_double_layers, model_info.num_single_layers)
                model._ts_ctx = StepContext(num_steps=model.num_steps)
                _dbg(f"cache_init: double={model_info.num_double_layers}, single={model_info.num_single_layers}, num_steps={model.num_steps}")

            cache_dic = model._ts_cache_dic
            ctx = model._ts_ctx

            strategy.schedule_step(cache_dic, ctx)
            step_t0 = time.time()
            _dbg(f"=== STEP {ctx.step} type={ctx.type} mem={_gpu_mem_mb():.0f}MB ===")

            # LoRA scale handling
            if attention_kwargs is not None:
                lora_scale = attention_kwargs.get('scale', 1.0)
            else:
                lora_scale = 1.0
            if USE_PEFT_BACKEND:
                scale_lora_layers(model, lora_scale)

            # HunyuanVideo-specific preprocessing
            batch_size, _num_channels, num_frames, height, width = hidden_states.shape
            p, p_t = model.config.patch_size, model.config.patch_size_t
            post_patch_num_frames = num_frames // p_t
            post_patch_height = height // p
            post_patch_width = width // p
            _dbg(f"input: B={batch_size} C={_num_channels} F={num_frames} H={height} W={width} patch=({p_t},{p})")

            t0 = time.time()
            image_rotary_emb = model.rope(hidden_states)
            _dbg(f"rope: {time.time()-t0:.3f}s")

            t0 = time.time()
            temb, _ = model.time_text_embed(timestep, pooled_projections, guidance)
            _dbg(f"time_text_embed: {time.time()-t0:.3f}s")

            t0 = time.time()
            hidden_states = model.x_embedder(hidden_states)
            _dbg(f"x_embedder: {time.time()-t0:.3f}s  hidden_states={tuple(hidden_states.shape)}")

            t0 = time.time()
            encoder_hidden_states = model.context_embedder(
                encoder_hidden_states, timestep, encoder_attention_mask
            )
            _dbg(f"context_embedder: {time.time()-t0:.3f}s  enc={tuple(encoder_hidden_states.shape)}")

            # Build attention mask [B, 1, 1, N]
            latent_seq_len = hidden_states.shape[1]
            cond_seq_len = encoder_hidden_states.shape[1]
            seq_len = latent_seq_len + cond_seq_len
            _dbg(f"seq_len: latent={latent_seq_len} cond={cond_seq_len} total={seq_len}")
            attn_mask = torch.ones(
                batch_size, seq_len, device=hidden_states.device, dtype=torch.bool
            )
            eff_cond_len = encoder_attention_mask.sum(dim=1, dtype=torch.int)
            eff_seq_len = latent_seq_len + eff_cond_len
            indices = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
            attn_mask = attn_mask.masked_fill(indices >= eff_seq_len.unsqueeze(1), False)
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N]
            _dbg(f"attn_mask: {tuple(attn_mask.shape)} mem={_gpu_mem_mb():.0f}MB")

            # Double stream blocks
            ctx.stream = 'double_stream'
            n_double = len(list(model.transformer_blocks))
            for idx, block in enumerate(model.transformer_blocks):
                ctx.layer = idx
                t0 = time.time()
                if ctx.type == 'full':
                    hidden_states, encoder_hidden_states = adapter.forward_double_block_full(
                        block, hidden_states, encoder_hidden_states,
                        temb, image_rotary_emb, strategy, cache_dic, ctx,
                        attention_mask=attn_mask,
                    )
                else:
                    hidden_states, encoder_hidden_states = adapter.forward_double_block_cached(
                        block, hidden_states, encoder_hidden_states,
                        temb, image_rotary_emb, strategy, cache_dic, ctx,
                    )
                strategy.on_block_end(cache_dic, ctx, hidden_states)
                _dbg(f"double[{idx}/{n_double}] {time.time()-t0:.3f}s mem={_gpu_mem_mb():.0f}MB")

            # Single stream blocks
            ctx.stream = 'single_stream'
            n_single = len(list(model.single_transformer_blocks))
            for idx, block in enumerate(model.single_transformer_blocks):
                ctx.layer = idx
                t0 = time.time()
                if ctx.type == 'full':
                    hidden_states, encoder_hidden_states = adapter.forward_single_block_full(
                        block, hidden_states, temb, image_rotary_emb,
                        strategy, cache_dic, ctx,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attn_mask,
                    )
                else:
                    hidden_states, encoder_hidden_states = adapter.forward_single_block_cached(
                        block, hidden_states, temb, image_rotary_emb,
                        strategy, cache_dic, ctx,
                        encoder_hidden_states=encoder_hidden_states,
                    )
                strategy.on_block_end(cache_dic, ctx, hidden_states)
                _dbg(f"single[{idx}/{n_single}] {time.time()-t0:.3f}s mem={_gpu_mem_mb():.0f}MB")

            # Output projection + video reshape
            t0 = time.time()
            hidden_states = model.norm_out(hidden_states, temb)
            hidden_states = model.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(
                batch_size, post_patch_num_frames, post_patch_height, post_patch_width,
                -1, p_t, p, p
            )
            hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
            hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
            _dbg(f"output_proj: {time.time()-t0:.3f}s  out={tuple(hidden_states.shape)}")

            if USE_PEFT_BACKEND:
                unscale_lora_layers(model, lora_scale)

            if hidden_states.dtype == torch.float16:
                hidden_states = hidden_states.clip(-65504, 65504)

            _dbg(f"=== STEP {ctx.step} DONE {time.time()-step_t0:.3f}s mem={_gpu_mem_mb():.0f}MB ===")
            ctx.step += 1

            # Free cache after last denoising step to reclaim VRAM for VAE decode
            if ctx.step >= ctx.num_steps:
                _dbg(f"Clearing cache after final step. mem_before={_gpu_mem_mb():.0f}MB")
                del model._ts_cache_dic, model._ts_ctx
                torch.cuda.empty_cache()
                _dbg(f"Cache cleared. mem_after={_gpu_mem_mb():.0f}MB")

            if not return_dict:
                return (hidden_states,)
            return Transformer2DModelOutput(sample=hidden_states)

        return patched_forward

    def forward_double_block_full(
        self, block, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor, rotary_emb, strategy: CacheStrategy,
        cache_dic: Dict, ctx: Any, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_mask = kwargs.get('attention_mask')

        t0 = time.time()
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(
            hidden_states, emb=temb
        )
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = \
            block.norm1_context(encoder_hidden_states, emb=temb)
        _dbg(f"  dbl_full[{ctx.layer}] norm: {time.time()-t0:.3f}s")

        t0 = time.time()
        attn_output, context_attn_output = block.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=rotary_emb,
        )
        _dbg(f"  dbl_full[{ctx.layer}] attn: {time.time()-t0:.3f}s  attn_out={tuple(attn_output.shape)} mem={_gpu_mem_mb():.0f}MB")

        ctx.module = 'img_attn'
        strategy.on_full_compute(cache_dic, ctx, attn_output)
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        t0 = time.time()
        ctx.module = 'img_mlp'
        norm_hidden_states = block.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = block.ff(norm_hidden_states)
        _dbg(f"  dbl_full[{ctx.layer}] ff: {time.time()-t0:.3f}s")
        strategy.on_full_compute(cache_dic, ctx, ff_output)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        ctx.module = 'txt_attn'
        strategy.on_full_compute(cache_dic, ctx, context_attn_output)
        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output

        t0 = time.time()
        ctx.module = 'txt_mlp'
        norm_encoder_hidden_states = block.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        context_ff_output = block.ff_context(norm_encoder_hidden_states)
        _dbg(f"  dbl_full[{ctx.layer}] ff_ctx: {time.time()-t0:.3f}s")
        strategy.on_full_compute(cache_dic, ctx, context_ff_output)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return hidden_states, encoder_hidden_states

    def forward_double_block_cached(
        self, block, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor, rotary_emb, strategy: CacheStrategy,
        cache_dic: Dict, ctx: Any, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del rotary_emb, kwargs  # unused in cached path
        _, gate_msa, _, _, gate_mlp = block.norm1(hidden_states, emb=temb)
        _, c_gate_msa, _, _, c_gate_mlp = block.norm1_context(encoder_hidden_states, emb=temb)

        ctx.module = 'img_attn'
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * strategy.on_cache_restore(cache_dic, ctx)

        ctx.module = 'img_mlp'
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * strategy.on_cache_restore(cache_dic, ctx)

        ctx.module = 'txt_attn'
        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * strategy.on_cache_restore(cache_dic, ctx)

        ctx.module = 'txt_mlp'
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * strategy.on_cache_restore(cache_dic, ctx)

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return hidden_states, encoder_hidden_states

    def forward_single_block_full(
        self, block, hidden_states: torch.Tensor, temb: torch.Tensor,
        rotary_emb, strategy: CacheStrategy, cache_dic: Dict, ctx: Any, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_hidden_states = kwargs['encoder_hidden_states']
        attention_mask = kwargs.get('attention_mask')
        text_seq_length = encoder_hidden_states.shape[1]

        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        residual = hidden_states

        t0 = time.time()
        norm_hidden_states, gate = block.norm(hidden_states, emb=temb)
        mlp_hidden_states = block.act_mlp(block.proj_mlp(norm_hidden_states))
        _dbg(f"  sgl_full[{ctx.layer}] norm+mlp: {time.time()-t0:.3f}s")

        norm_img = norm_hidden_states[:, :-text_seq_length, :]
        norm_txt = norm_hidden_states[:, -text_seq_length:, :]

        t0 = time.time()
        attn_output, context_attn_output = block.attn(
            hidden_states=norm_img,
            encoder_hidden_states=norm_txt,
            attention_mask=attention_mask,
            image_rotary_emb=rotary_emb,
        )
        _dbg(f"  sgl_full[{ctx.layer}] attn: {time.time()-t0:.3f}s mem={_gpu_mem_mb():.0f}MB")

        attn_output = torch.cat([attn_output, context_attn_output], dim=1)
        proj_input = torch.cat([attn_output, mlp_hidden_states], dim=2)
        proj_output = block.proj_out(proj_input)

        ctx.module = 'total'
        strategy.on_full_compute(cache_dic, ctx, proj_output)

        hidden_states = residual + gate.unsqueeze(1) * proj_output

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states[:, :-text_seq_length, :], hidden_states[:, -text_seq_length:, :]

    def forward_single_block_cached(
        self, block, hidden_states: torch.Tensor, temb: torch.Tensor,
        rotary_emb, strategy: CacheStrategy, cache_dic: Dict, ctx: Any, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del rotary_emb  # unused in cached path
        encoder_hidden_states = kwargs['encoder_hidden_states']
        text_seq_length = encoder_hidden_states.shape[1]

        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        residual = hidden_states

        _, gate = block.norm(hidden_states, emb=temb)

        ctx.module = 'total'
        proj_output = strategy.on_cache_restore(cache_dic, ctx)

        hidden_states = residual + gate.unsqueeze(1) * proj_output

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states[:, :-text_seq_length, :], hidden_states[:, -text_seq_length:, :]
