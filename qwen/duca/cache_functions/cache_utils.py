import torch
import torch.nn.functional as F


def force_init(cache_dic, current, tokens):
    '''
    Initialization for Force Activation step.
    '''
    cache_dic['cache_index'][-1][current['stream']][current['layer']][current['module']] = torch.zeros(tokens.shape[0], tokens.shape[1], dtype=torch.int, device=tokens.device)


def update_cache(fresh_indices, fresh_tokens, cache_dic, current):
    '''
    Update the cache with the fresh tokens.
    '''
    indices = fresh_indices
    cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].scatter_(dim=1, index=indices.unsqueeze(-1).expand(-1, -1, fresh_tokens.shape[-1]), src=fresh_tokens)


def cache_cutfresh(cache_dic, tokens, current):
    '''
    Cut fresh tokens from the input tokens and update the cache counter.
    
    cache_dic: dict, the cache dictionary containing cache(main extra memory cost), indices and some other information.
    tokens: torch.Tensor, the input tokens to be cut.
    current: dict, the current step, layer, and module information. Particularly convenient for debugging.
    '''
    stream = current['stream']
    layer = current['layer']
    module = current['module']
    
    fresh_ratio = fresh_ratio_scheduler(cache_dic, current)
    fresh_ratio = torch.clamp(torch.tensor(fresh_ratio, device = tokens.device), min=0, max=1)
    
    # Generate the index tensor for fresh tokens
    score = score_evaluate(cache_dic, tokens, current) # s1, s2, s3 mentioned in the paper
    indices = score.argsort(dim=-1, descending=True)
    topk = int(fresh_ratio * score.shape[1])
    fresh_indices = indices[:, :topk]

    # Updating the Cache Frequency Score s3 mentioned in the paper
    # stale tokens index + 1 in each ***module***, fresh tokens index = 0
    cache_dic['cache_index'][-1][stream][layer][module] += 1
    cache_dic['cache_index'][-1][stream][layer][module].scatter_(dim=1, index=fresh_indices, src = torch.zeros_like(fresh_indices, dtype=torch.int, device=fresh_indices.device))
    
    fresh_indices_expand = fresh_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])

    fresh_tokens = torch.gather(input = tokens, dim = 1, index = fresh_indices_expand)

    return fresh_indices, fresh_tokens


def fresh_ratio_scheduler(cache_dic, current):
    '''
    Return the fresh ratio for the current step.
    '''
    fresh_ratio = cache_dic['fresh_ratio']
    step = current['step']
    num_steps = cache_dic['num_steps']

    step_weight = 2.0
    step_factor = 1 + step_weight - 2 * step_weight * step / num_steps

    layer_weight = - 0.2
    layer_factor = 1 + layer_weight - 2 * layer_weight * current['layer'] / 59

    # if you want worse performance, you can use the following setting
    # step_weight = 0.0
    # step_factor = 1 - step_weight + 2 * step_weight * step / num_steps

    # layer_weight = 0.5
    # layer_factor = 1 + layer_weight - 2 * layer_weight * current['layer'] / 59

    module_weight = 2.5 # no calculations for attn module (2.5 * 0.4 = 1.0), compuation is transformed to mlp module.
    module_time_weight = 0.6 # estimated from the time and flops of mlp and attn module, may change in different situations.
    module_factor = 1 + module_time_weight * module_weight # for mlp
    
    return fresh_ratio * layer_factor * step_factor * module_factor


def score_evaluate(cache_dic, tokens, current) -> torch.Tensor:
    '''
    Return the score tensor (B, N) for the given tokens.
    '''
    # Just see more explanation in the version of DiT-ToCa if needed.
    score = torch.rand(tokens.shape[0], tokens.shape[1], device=tokens.device)

    soft_step_score = cache_dic['cache_index'][-1][current['stream']][current['layer']][current['module']].float() / (cache_dic['interval'])
    score = score + cache_dic['soft_fresh_weight'] * soft_step_score 
    
    return score.to(tokens.device)


def pipeline_with_cache(pipe):

    import types
    from pipeline.transformer_qwenimage import QwenImageTransformer2DModel as LocalQwenImageTransformer2DModel
    from pipeline.transformer_qwenimage import QwenImageTransformerBlock as LocalQwenImageTransformerBlock

    pipe.transformer.forward = types.MethodType(LocalQwenImageTransformer2DModel.forward, pipe.transformer)

    for _, block in enumerate(pipe.transformer.transformer_blocks):
        block.forward = types.MethodType(LocalQwenImageTransformerBlock.forward, block)
        
    return pipe