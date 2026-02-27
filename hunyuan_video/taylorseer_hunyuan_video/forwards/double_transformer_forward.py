import torch
from typing import Any, Dict, Optional, Tuple, Union
from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoTransformerBlock

from taylorseer_core import module_cache_init as taylor_cache_init
from taylorseer_core.forward_utils import update_cache_or_approximate


def taylorseer_hunyuan_video_double_block_forward(
    self: HunyuanVideoTransformerBlock,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.FloatTensor,
    attention_mask: Optional[torch.Tensor] = None,
    freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
        encoder_hidden_states, emb=temb
    )

    joint_attention_kwargs = joint_attention_kwargs or {}
    cache_dic = joint_attention_kwargs['cache_dic']
    current = joint_attention_kwargs['current']

    if current['type'] == 'full':
        # Placeholder init for combined attn slot
        current['module'] = 'attn'
        taylor_cache_init(cache_dic=cache_dic, current=current)

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
        )

        current['module'] = 'img_attn'
        update_cache_or_approximate(cache_dic, current, attn_output)
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        current['module'] = 'img_mlp'
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)
        update_cache_or_approximate(cache_dic, current, ff_output)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        current['module'] = 'txt_attn'
        update_cache_or_approximate(cache_dic, current, context_attn_output)
        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output

        current['module'] = 'txt_mlp'
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        update_cache_or_approximate(cache_dic, current, context_ff_output)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

    elif current['type'] == 'Taylor':
        current['module'] = 'img_attn'
        attn_output = update_cache_or_approximate(cache_dic, current, None)
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        current['module'] = 'img_mlp'
        ff_output = update_cache_or_approximate(cache_dic, current, None)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output

        current['module'] = 'txt_attn'
        context_attn_output = update_cache_or_approximate(cache_dic, current, None)
        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output

        current['module'] = 'txt_mlp'
        context_ff_output = update_cache_or_approximate(cache_dic, current, None)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return hidden_states, encoder_hidden_states
