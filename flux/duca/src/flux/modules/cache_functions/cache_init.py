def cache_init(**kwargs):   
    '''
    Initialization for cache.
    '''
    cache = {}
    cache[-1]={}
    cache[-1]['double_stream']={}
    cache[-1]['single_stream']={}
    cache_index = {}
    cache_index[-1]={}
    cache_dic = {}
    cache_dic['attn_map'] = {}
    cache_dic['attn_map'][-1] = {}
    cache_dic['attn_map'][-1]['double_stream'] = {}
    cache_dic['attn_map'][-1]['single_stream'] = {}

    for i in range(19):
        cache[-1]['double_stream'][i] = {}
        cache_index[-1][i] = {}
        cache_dic['attn_map'][-1]['double_stream'][i] = {}
        cache_dic['attn_map'][-1]['double_stream'][i]['total'] = {}
        cache_dic['attn_map'][-1]['double_stream'][i]['txt_mlp'] = {}
        cache_dic['attn_map'][-1]['double_stream'][i]['img_mlp'] = {}

    for i in range(38):
        cache[-1]['single_stream'][i] = {}
        cache_index[-1][i] = {}
        cache_dic['attn_map'][-1]['single_stream'][i] = {}
        cache_dic['attn_map'][-1]['single_stream'][i]['total'] = {}

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
