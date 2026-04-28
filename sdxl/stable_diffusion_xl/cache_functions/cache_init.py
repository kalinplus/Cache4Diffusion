def cache_init(**kwargs):   
    '''
    Initialization for cache.
    '''
    cache = {}
    cache_dic = {}
    cache_dic['cache'] = cache
    cache_dic['height'] = kwargs['height']
    cache_dic['width'] = kwargs['width']
    cache_dic['num_steps'] = kwargs['num_steps']
    cache_dic['test_FLOPs'] = kwargs['test_FLOPs']
    cache_dic['monitor_gpu_usage'] = kwargs['monitor_gpu_usage']

    current = {}

    return cache_dic, current

