def cache_init(kwargs):   
    '''
    Initialization for cache.
    '''
    cache = {}
    cache[-1]={}
    cache_index = {}
    cache_index[-1]={}
    cache_dic = {}
    cache_dic['attn_map'] = {}
    cache_dic['attn_map'][-1] = {}

    for stream in ['cond', 'uncond']:
        cache[-1][stream] = {}
        cache_index[-1][stream] = {}
        cache_dic['attn_map'][-1][stream] = {}

        for layer_idx in range(60):
            cache[-1][stream][layer_idx] = {}
            cache_index[-1][stream][layer_idx] = {}
            cache_dic['attn_map'][-1][stream][layer_idx] = {}
            cache_dic['attn_map'][-1][stream][layer_idx]['total'] = {}
            cache_dic['attn_map'][-1][stream][layer_idx]['txt_mlp'] = {}
            cache_dic['attn_map'][-1][stream][layer_idx]['img_mlp'] = {}

    cache_dic['cache'] = cache
    cache_dic['cache_index'] = cache_index
    cache_dic['num_steps'] = kwargs['num_steps']
    cache_dic['test_FLOPs'] = kwargs['test_FLOPs']
    cache_dic['monitor_gpu_usage'] = kwargs['monitor_gpu_usage']
    cache_dic['interval'] = kwargs['interval']
    cache_dic['first_enhance'] = kwargs['first_enhance']
    cache_dic['fresh_ratio'] = kwargs['fresh_ratio']
    cache_dic['soft_fresh_weight'] = kwargs['soft_fresh_weight']
    
    current = {}
    current['step'] = 0
    current['cache_counter'] = 0

    return cache_dic, current
