# Updated QwenImageTransformerBlock forward with taylorseer and proper cond/uncond support
# Based on the correct implementation pattern

import os
import torch
from typing import Any, Dict, Optional, Tuple
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock
from qwen_image.taylorseer_qwen_image.taylorseer_utils import (
    derivative_approximation,
    taylor_formula,
    module_cache_init,
    shift_cache_history,
    derivative_approximation_with_smoothing
)

_TS_DEBUG_SHAPES = os.environ.get("TS_DEBUG_SHAPES", "0").lower() in ("1", "true", "yes")

def taylorseer_qwen_image_mmdit_forward(
    self: QwenImageTransformerBlock,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_mask: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    cache_dic: Optional[Dict[str, Any]] = None,
    current: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Updated forward method that properly handles positive/negative prompts with TaylorSeer caching.
    """
    # cache_dic 和 current 现在直接作为参数传入

    # Check if this is a full computation or Taylor approximation
    if current['type'] == 'full':
        # Get modulation parameters for both streams
        img_mod_params = self.img_mod(temb)  # [B, 6*dim]
        txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]

        # Split modulation parameters for norm1 and norm2
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

        # Process image stream - norm1 + modulation
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

        # Process text stream - norm1 + modulation
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        # Joint attention computation
        attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
        )
        img_attn_output, txt_attn_output = attn_output

        # Handle image attention with caching
        current['module'] = 'img_attn'
        module_cache_init(cache_dic=cache_dic, current=current)
        if cache_dic.get('use_smoothing', False):
            shift_cache_history(cache_dic=cache_dic, current=current)
            derivative_approximation_with_smoothing(
                cache_dic=cache_dic,
                current=current,
                feature=img_attn_output,
                smoothing_method=cache_dic.get('smoothing_method', 'exponential'),
                alpha=cache_dic.get('smoothing_alpha', 0.8)
            )
        else:
            derivative_approximation(cache_dic=cache_dic, current=current, feature=img_attn_output)
        hidden_states = hidden_states + img_gate1 * img_attn_output

        # Handle text attention with caching
        current['module'] = 'txt_attn'
        module_cache_init(cache_dic=cache_dic, current=current)
        if cache_dic.get('use_smoothing', False):
            shift_cache_history(cache_dic=cache_dic, current=current)
            derivative_approximation_with_smoothing(
                cache_dic=cache_dic,
                current=current,
                feature=txt_attn_output,
                smoothing_method=cache_dic.get('smoothing_method', 'exponential'),
                alpha=cache_dic.get('smoothing_alpha', 0.8)
            )
        else:
            derivative_approximation(cache_dic=cache_dic, current=current, feature=txt_attn_output)
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # Process image stream - norm2 + MLP
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # Handle image MLP with caching
        current['module'] = 'img_mlp'
        module_cache_init(cache_dic=cache_dic, current=current)
        if cache_dic.get('use_smoothing', False):
            shift_cache_history(cache_dic=cache_dic, current=current)
            derivative_approximation_with_smoothing(
                cache_dic=cache_dic,
                current=current,
                feature=img_mlp_output,
                smoothing_method=cache_dic.get('smoothing_method', 'exponential'),
                alpha=cache_dic.get('smoothing_alpha', 0.8)
            )
        else:
            derivative_approximation(cache_dic=cache_dic, current=current, feature=img_mlp_output)

        # Process text stream - norm2 + MLP
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        # Handle text MLP with caching
        current['module'] = 'txt_mlp'
        module_cache_init(cache_dic=cache_dic, current=current)
        if cache_dic.get('use_smoothing', False):
            shift_cache_history(cache_dic=cache_dic, current=current)
            derivative_approximation_with_smoothing(
                cache_dic=cache_dic,
                current=current,
                feature=txt_mlp_output,
                smoothing_method=cache_dic.get('smoothing_method', 'exponential'),
                alpha=cache_dic.get('smoothing_alpha', 0.8)
            )
        else:
            derivative_approximation(cache_dic=cache_dic, current=current, feature=txt_mlp_output)

    else:
        # Taylor approximation mode - use cached values
        img_mod_params = self.img_mod(temb)
        txt_mod_params = self.txt_mod(temb)

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

        # Apply Taylor approximation for each module
        current['module'] = 'img_attn'
        hidden_states = hidden_states + img_gate1 * taylor_formula(cache_dic=cache_dic, current=current)

        current['module'] = 'txt_attn'
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * taylor_formula(cache_dic=cache_dic, current=current)

        current['module'] = 'img_mlp'
        hidden_states = hidden_states + img_gate2 * taylor_formula(cache_dic=cache_dic, current=current)

        current['module'] = 'txt_mlp'
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * taylor_formula(cache_dic=cache_dic, current=current)

    # Clip to prevent overflow for fp16
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states