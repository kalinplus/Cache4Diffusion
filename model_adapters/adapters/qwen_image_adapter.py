"""
QwenImage model adapter implementation.
QwenImage only has double-stream transformer blocks (no single stream).
QwenImage uses true CFG: the pipeline calls forward twice per step
(once for cond/prompt, once for uncond/negative prompt).
Cache is branched by 'cond'/'uncond' instead of 'double_stream'/'single_stream'.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

from model_adapters.base import ModelAdapter
from model_adapters.info import ModelInfo
from caching_core import CacheStrategy


class QwenImageAdapter(ModelAdapter):
    """QwenImage model adapter. Only double-stream blocks, no single stream.
    Uses dual cond/uncond cache branches for true CFG."""

    def get_model_info(self, model) -> ModelInfo:
        return ModelInfo(
            num_double_layers=model.config.num_layers,
            num_single_layers=0,
            has_double_stream=True,
            has_single_stream=False,
        )

    def get_block_iterators(self, model) -> Dict[str, List]:
        return {
            'double_stream': list(model.transformer_blocks),
        }

    def create_forward_fn(self, model, strategy: CacheStrategy):
        """QwenImage-specific forward with dual cond/uncond cache branches.

        The diffusers pipeline wraps each forward call in
        ``model.cache_context("cond")`` or ``model.cache_context("uncond")``.
        We intercept that context manager to set ``model._ts_cache_branch``
        so the patched forward knows which cache branch to use — no fragile
        toggle or call-order assumptions needed.

        When ``do_true_cfg=False`` (no negative prompt), the pipeline only
        calls forward once per step under ``cache_context("cond")``, and
        the step increments normally.
        """
        model_info = self.get_model_info(model)
        adapter = self
        num_layers = model_info.num_double_layers

        # --- Wrap cache_context to expose branch name -----------------------
        from contextlib import contextmanager

        _original_cache_context = model.cache_context
        model._ts_cache_branch = None  # sentinel

        @contextmanager
        def _wrapped_cache_context(name: str):
            model._ts_cache_branch = name
            with _original_cache_context(name):
                yield
            model._ts_cache_branch = None

        model.cache_context = _wrapped_cache_context

        # --- Closure state --------------------------------------------------
        state = {'cache_dic': None, 'current': None, 'last_branch': None}

        def patched_forward(
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            encoder_hidden_states_mask: torch.Tensor = None,
            timestep: torch.LongTensor = None,
            img_shapes=None,
            txt_seq_lens=None,
            guidance: torch.Tensor = None,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            controlnet_block_samples=None,
            return_dict: bool = True,
            cache_dic: Optional[Dict[str, Any]] = None,
            current: Optional[Dict[str, Any]] = None,
        ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:

            # Initialize cache on first call with cond/uncond branches
            if state['cache_dic'] is None:
                from taylorseer_core import cache_init
                from caching_core import StepContext
                state['cache_dic'] = cache_init(
                    branches={'cond': num_layers, 'uncond': num_layers}
                )
                state['current'] = StepContext(num_steps=model.num_steps)

            cache_dic = state['cache_dic']
            current = state['current']

            # Read branch from diffusers cache_context; default to 'cond'
            branch = getattr(model, '_ts_cache_branch', None) or 'cond'
            current.stream = branch

            # Schedule step type only on the cond (first) call of each step
            if branch == 'cond':
                strategy.schedule_step(cache_dic, current)

            # LoRA scale handling
            if attention_kwargs is not None:
                attention_kwargs = attention_kwargs.copy()
                lora_scale = attention_kwargs.pop("scale", 1.0)
            else:
                lora_scale = 1.0

            if USE_PEFT_BACKEND:
                scale_lora_layers(model, lora_scale)

            # QwenImage-specific embeddings
            hidden_states = model.img_in(hidden_states)

            timestep = timestep.to(hidden_states.dtype)
            encoder_hidden_states = model.txt_norm(encoder_hidden_states)
            encoder_hidden_states = model.txt_in(encoder_hidden_states)

            if guidance is not None:
                guidance = guidance.to(hidden_states.dtype) * 1000

            temb = (
                model.time_text_embed(timestep, hidden_states)
                if guidance is None
                else model.time_text_embed(timestep, guidance, hidden_states)
            )

            image_rotary_emb = model.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

            # Double stream blocks (QwenImage has no single stream)
            for idx, block in enumerate(model.transformer_blocks):
                current.layer = idx
                if current.type == 'full':
                    encoder_hidden_states, hidden_states = adapter.forward_double_block_full(
                        block, hidden_states, encoder_hidden_states,
                        temb, image_rotary_emb, strategy, cache_dic, current,
                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                        attention_kwargs=attention_kwargs,
                    )
                else:
                    encoder_hidden_states, hidden_states = adapter.forward_double_block_cached(
                        block, hidden_states, encoder_hidden_states,
                        temb, image_rotary_emb, strategy, cache_dic, current,
                    )

                strategy.on_block_end(cache_dic, current, hidden_states)

                if controlnet_block_samples is not None:
                    interval = int(np.ceil(len(model.transformer_blocks) / len(controlnet_block_samples)))
                    hidden_states = hidden_states + controlnet_block_samples[idx // interval]

            hidden_states = model.norm_out(hidden_states, temb)
            output = model.proj_out(hidden_states)

            if USE_PEFT_BACKEND:
                unscale_lora_layers(model, lora_scale)

            # Increment step after the last branch call of each timestep.
            # With true CFG: cond -> uncond -> step++
            # Without true CFG: cond -> step++
            prev_branch = state['last_branch']
            state['last_branch'] = branch
            if branch == 'uncond':
                # uncond is always the last call in a true-CFG step
                current.step += 1
            elif branch == 'cond' and prev_branch != 'uncond':
                # No uncond followed the previous cond -> single-branch mode
                # (first step is always cond with prev=None, so skip it)
                if prev_branch is not None:
                    current.step += 1

            if not return_dict:
                return (output,)
            return Transformer2DModelOutput(sample=output)

        return patched_forward

    def forward_double_block_full(
        self, block, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor, rotary_emb, strategy: CacheStrategy,
        cache_dic: Dict, ctx: Any, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward for QwenImage double transformer block."""
        encoder_hidden_states_mask = kwargs.get('encoder_hidden_states_mask')
        attention_kwargs = kwargs.get('attention_kwargs') or {}

        img_mod_params = block.img_mod(temb)
        txt_mod_params = block.txt_mod(temb)
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        img_normed = block.img_norm1(hidden_states)
        img_modulated, img_gate1 = block._modulate(img_normed, img_mod1)

        txt_normed = block.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = block._modulate(txt_normed, txt_mod1)

        attn_output = block.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=rotary_emb,
            **attention_kwargs,
        )
        img_attn_output, txt_attn_output = attn_output

        ctx.module = 'img_attn'
        strategy.on_full_compute(cache_dic, ctx, img_attn_output)
        hidden_states = hidden_states + img_gate1 * img_attn_output

        ctx.module = 'txt_attn'
        strategy.on_full_compute(cache_dic, ctx, txt_attn_output)
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        img_normed2 = block.img_norm2(hidden_states)
        img_modulated2, img_gate2 = block._modulate(img_normed2, img_mod2)
        img_mlp_output = block.img_mlp(img_modulated2)

        ctx.module = 'img_mlp'
        strategy.on_full_compute(cache_dic, ctx, img_mlp_output)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        txt_normed2 = block.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = block._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = block.txt_mlp(txt_modulated2)

        ctx.module = 'txt_mlp'
        strategy.on_full_compute(cache_dic, ctx, txt_mlp_output)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    def forward_double_block_cached(
        self, block, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor, rotary_emb, strategy: CacheStrategy,
        cache_dic: Dict, ctx: Any, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cache restore for QwenImage double transformer block."""
        img_mod_params = block.img_mod(temb)
        txt_mod_params = block.txt_mod(temb)
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        _, _, img_gate1 = img_mod1.chunk(3, dim=-1)
        _, _, img_gate2 = img_mod2.chunk(3, dim=-1)
        _, _, txt_gate1 = txt_mod1.chunk(3, dim=-1)
        _, _, txt_gate2 = txt_mod2.chunk(3, dim=-1)

        img_gate1 = img_gate1.unsqueeze(1)
        img_gate2 = img_gate2.unsqueeze(1)
        txt_gate1 = txt_gate1.unsqueeze(1)
        txt_gate2 = txt_gate2.unsqueeze(1)

        ctx.module = 'img_attn'
        hidden_states = hidden_states + img_gate1 * strategy.on_cache_restore(cache_dic, ctx)

        ctx.module = 'txt_attn'
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * strategy.on_cache_restore(cache_dic, ctx)

        ctx.module = 'img_mlp'
        hidden_states = hidden_states + img_gate2 * strategy.on_cache_restore(cache_dic, ctx)

        ctx.module = 'txt_mlp'
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * strategy.on_cache_restore(cache_dic, ctx)

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    def forward_single_block_full(self, block, hidden_states, temb, rotary_emb,
                                   strategy, cache_dic, ctx, **kwargs):
        raise NotImplementedError("QwenImage has no single stream blocks")

    def forward_single_block_cached(self, block, hidden_states, temb, rotary_emb,
                                     strategy, cache_dic, ctx, **kwargs):
        raise NotImplementedError("QwenImage has no single stream blocks")
