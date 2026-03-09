def cache_init(**kwargs):   
    '''
    Initialization for cache.
    '''
    cache = {}
    cache[-1]={}

    cache_dic = {}
    cache_dic['cache'] = cache
    cache_dic['num_steps'] = kwargs['num_steps']
    cache_dic['test_FLOPs'] = kwargs['test_FLOPs']
    cache_dic['monitor_gpu_usage'] = kwargs['monitor_gpu_usage']
    cache_dic['enable_teacache'] = kwargs['enable_teacache']
    cache_dic['rel_l1_thresh'] = kwargs['rel_l1_thresh']
    cache_dic['coefficients'] = kwargs['coefficients']

    current = {}
    current['cnt'] = 0
    current['accumulated_rel_l1_distance'] = 0
    current['previous_modulated_input'] = None
    current['previous_residual'] = None

    return cache_dic, current

