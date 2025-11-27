import torch
from typing import Any, Dict, Optional, Tuple, Union
from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock
from taylorseer_utils import (
    derivative_approximation,
    derivative_approximation_with_smoothing,
    derivative_approximation_hybrid_smoothing,
    shift_cache_history,
    taylor_formula, 
    taylor_cache_init
)

def taylorseer_flux_single_block_forward(
    self: FluxSingleTransformerBlock,
    hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    image_rotary_emb=None,
    joint_attention_kwargs=None,
):
    joint_attention_kwargs = joint_attention_kwargs or {}
    cache_dic = joint_attention_kwargs['cache_dic']
    current = joint_attention_kwargs['current']
    
    # 获取平滑配置
    use_smoothing = cache_dic.get('use_smoothing', False)
    smoothing_method = cache_dic.get('smoothing_method', 'moving_average')
    smoothing_alpha = cache_dic.get('smoothing_alpha', 0.7)
    use_hybrid_smoothing = cache_dic.get('use_hybrid_smoothing', False)

    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    gate = gate.unsqueeze(1)

    residual = hidden_states
    
    if current['type'] == 'full':

        current['module'] = 'total'
        if use_smoothing:
            shift_cache_history(cache_dic=cache_dic, current=current)
        taylor_cache_init(cache_dic=cache_dic, current=current)

        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            #**joint_attention_kwargs,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)

        hidden_states = self.proj_out(hidden_states)
        
        if use_smoothing and use_hybrid_smoothing:
            # print("[INFO] Using hybrid smoothing for image attention in single stream.")
            derivative_approximation_hybrid_smoothing(
                cache_dic=cache_dic, 
                current=current, 
                feature=hidden_states,
                smoothing_method=smoothing_method,
                alpha=smoothing_alpha
            )
        elif use_smoothing:
            # print("[INFO] Using smoothing for image attention in single stream.")
            derivative_approximation_with_smoothing(
                cache_dic=cache_dic, 
                current=current, 
                feature=hidden_states,
                smoothing_method=smoothing_method,
                alpha=smoothing_alpha
            )
        else:
            # print("[IFNO] No smoothing for image attention in single stream.")
            derivative_approximation(cache_dic=cache_dic, current=current, feature=hidden_states)


    elif current['type'] == 'Taylor':

        current['module'] = 'total'
        hidden_states = taylor_formula(cache_dic=cache_dic, current=current)

    hidden_states = gate * hidden_states
    hidden_states = residual + hidden_states

    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    return hidden_states