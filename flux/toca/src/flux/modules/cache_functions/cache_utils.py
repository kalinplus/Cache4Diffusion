import torch
import torch.nn.functional as F


def force_init(cache_dic, current, tokens):
    '''
    Initialization for Force Activation step.
    '''
    cache_dic['cache_index'][-1][current['layer']][current['module']] = torch.zeros(tokens.shape[0], tokens.shape[1], dtype=torch.int, device=tokens.device)


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
    cache_dic['cache_index'][-1][layer][module] += 1
    cache_dic['cache_index'][-1][layer][module].scatter_(dim=1, index=fresh_indices, src = torch.zeros_like(fresh_indices, dtype=torch.int, device=fresh_indices.device))
    
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
    layer_bound = 37 if current['stream'] == 'double_stream' else 18

    step_weight = 0.0
    step_factor = 1 - step_weight + 2 * step_weight * step / num_steps

    layer_weight = 0.5
    layer_factor = 1 + layer_weight - 2 * layer_weight * current['layer'] / layer_bound

    stream_weight = 0.6
    stream_factor = (1 - stream_weight) if current['stream']=='double_stream' else (1 + stream_weight)
    
    return fresh_ratio * layer_factor * step_factor * stream_factor


def score_evaluate(cache_dic, tokens, current) -> torch.Tensor:
    '''
    Return the score tensor (B, N) for the given tokens.
    '''
    # Just see more explanation in the version of DiT-ToCa if needed.
    if current['stream'] == 'double_stream':
        score = F.normalize(cache_dic['attn_map'][-1][current['stream']][current['layer']][current['module']], dim=-1, p=2)
    elif current['stream'] == 'single_stream':
        score = F.normalize(cache_dic['attn_map'][-1][current['stream']][current['layer']]['total'], dim=-1, p=2)

    soft_step_score = cache_dic['cache_index'][-1][current['layer']][current['module']].float() / (cache_dic['interval'])
    score = score + cache_dic['soft_fresh_weight'] * soft_step_score 
    
    return score.to(tokens.device)