import os
import torch
from typing import Any, Dict, Optional, Tuple
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock

from taylorseer_core.forward_utils import update_cache_or_approximate

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
    Forward method with TaylorSeer caching for QwenImage MMDiT blocks.
    Uses update_cache_or_approximate for unified cache/Taylor logic.
    Note: order unified to shift_cache_history -> module_cache_init (same as Flux).
    """
    if current['type'] == 'full':
        img_mod_params = self.img_mod(temb)
        txt_mod_params = self.txt_mod(temb)

        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
        )
        img_attn_output, txt_attn_output = attn_output

        current['module'] = 'img_attn'
        update_cache_or_approximate(cache_dic, current, img_attn_output)
        hidden_states = hidden_states + img_gate1 * img_attn_output

        current['module'] = 'txt_attn'
        update_cache_or_approximate(cache_dic, current, txt_attn_output)
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        img_mlp_output = self.img_mlp(img_modulated2)

        current['module'] = 'img_mlp'
        update_cache_or_approximate(cache_dic, current, img_mlp_output)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)

        current['module'] = 'txt_mlp'
        update_cache_or_approximate(cache_dic, current, txt_mlp_output)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

    else:
        # Taylor approximation mode
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

        current['module'] = 'img_attn'
        hidden_states = hidden_states + img_gate1 * update_cache_or_approximate(cache_dic, current, None)

        current['module'] = 'txt_attn'
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * update_cache_or_approximate(cache_dic, current, None)

        current['module'] = 'img_mlp'
        hidden_states = hidden_states + img_gate2 * update_cache_or_approximate(cache_dic, current, None)

        current['module'] = 'txt_mlp'
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * update_cache_or_approximate(cache_dic, current, None)

    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states
