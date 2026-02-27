import torch
from typing import Any, Dict, Optional, Tuple, Union
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock

from taylorseer_core import shift_cache_history, module_cache_init as taylor_cache_init
from taylorseer_core.forward_utils import update_cache_or_approximate


def taylorseer_flux_double_block_forward(
    self: FluxTransformerBlock,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    image_rotary_emb=None,
    joint_attention_kwargs=None,
):
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
        encoder_hidden_states, emb=temb
    )
    joint_attention_kwargs = joint_attention_kwargs or {}

    cache_dic = joint_attention_kwargs['cache_dic']
    current = joint_attention_kwargs['current']

    if current['type'] == 'full':
        # Placeholder init for the combined attn slot (kept for compatibility)
        current['module'] = 'attn'
        if cache_dic.get('use_smoothing', False):
            shift_cache_history(cache_dic=cache_dic, current=current)
        taylor_cache_init(cache_dic=cache_dic, current=current)

        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs
            raise NotImplementedError("Not implemented for TaylorSeer yet.")

        # ---- img_attn ----
        current['module'] = 'img_attn'
        update_cache_or_approximate(cache_dic, current, attn_output)
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        # ---- img_mlp ----
        current['module'] = 'img_mlp'
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)
        update_cache_or_approximate(cache_dic, current, ff_output)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        # ---- txt_attn ----
        current['module'] = 'txt_attn'
        update_cache_or_approximate(cache_dic, current, context_attn_output)
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        # ---- txt_mlp ----
        current['module'] = 'txt_mlp'
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        update_cache_or_approximate(cache_dic, current, context_ff_output)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    elif current['type'] == 'Taylor':
        current['module'] = 'img_attn'
        attn_output = update_cache_or_approximate(cache_dic, current, None)
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        current['module'] = 'img_mlp'
        ff_output = update_cache_or_approximate(cache_dic, current, None)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        current['module'] = 'txt_attn'
        context_attn_output = update_cache_or_approximate(cache_dic, current, None)
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        current['module'] = 'txt_mlp'
        context_ff_output = update_cache_or_approximate(cache_dic, current, None)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states
