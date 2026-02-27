import torch
from typing import Any, Dict, Optional, Tuple, Union
from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoSingleTransformerBlock

from taylorseer_core.forward_utils import update_cache_or_approximate


def taylorseer_hunyuan_video_single_block_forward(
    self: HunyuanVideoSingleTransformerBlock,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    joint_attention_kwargs = joint_attention_kwargs or {}
    cache_dic = joint_attention_kwargs['cache_dic']
    current = joint_attention_kwargs['current']

    text_seq_length = encoder_hidden_states.shape[1]
    hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
    residual = hidden_states

    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

    norm_hidden_states, norm_encoder_hidden_states = (
        norm_hidden_states[:, :-text_seq_length, :],
        norm_hidden_states[:, -text_seq_length:, :],
    )

    current['module'] = 'total'

    if current['type'] == 'full':
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        attn_output = torch.cat([attn_output, context_attn_output], dim=1)
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states = gate.unsqueeze(1) * self.proj_out(hidden_states)
        hidden_states = hidden_states + residual

        # Cache the full post-residual output (matches original behavior)
        update_cache_or_approximate(cache_dic, current, hidden_states)

        hidden_states, encoder_hidden_states = (
            hidden_states[:, :-text_seq_length, :],
            hidden_states[:, -text_seq_length:, :],
        )

    elif current['type'] == 'Taylor':
        hidden_states = update_cache_or_approximate(cache_dic, current, None)

        hidden_states, encoder_hidden_states = (
            hidden_states[:, :-text_seq_length, :],
            hidden_states[:, -text_seq_length:, :],
        )

    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return hidden_states, encoder_hidden_states
