def cache_init(kwargs):   
    '''
    Initialization for cache.
    '''
    cache = {}
    cache[-1] = {}
    cache[-2] = {}

    for stream in ['cond', 'uncond']:
        cache[-1][stream] = {}
        cache[-2][stream] = {}

        for layer_idx in range(60):
            cache[-1][stream][layer_idx] = {}
            cache[-2][stream][layer_idx] = {}

    cache_dic = {}
    cache_dic['cache'] = cache
    cache_dic['num_steps'] = kwargs['num_steps']
    cache_dic['test_FLOPs'] = kwargs['test_FLOPs']
    cache_dic['monitor_gpu_usage'] = kwargs['monitor_gpu_usage']
    cache_dic['interval'] = kwargs['interval']
    cache_dic['max_order'] = kwargs['max_order']
    cache_dic['first_enhance'] = kwargs['first_enhance']
    cache_dic['use_smoothing'] = kwargs.get('use_smoothing', False)
    cache_dic['smoothing_method'] = kwargs.get('smoothing_method', 'exponential')
    cache_dic['smoothing_alpha'] = kwargs.get('smoothing_alpha', 0.9)

    current = {}
    current['step'] = 0
    current['activated_steps'] = [0]
    current['cache_counter'] = 0

    return cache_dic, current

