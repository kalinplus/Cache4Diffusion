def cache_init(**kwargs):   
    '''
    Initialization for cache.
    '''
    cache = {}
    cache[-1]={}
    cache[-1]['double_stream']={}
    cache[-1]['single_stream']={}
    
    for i in range(19):
        cache[-1]['double_stream'][i] = {}

    for i in range(38):
        cache[-1]['single_stream'][i] = {}

    cache_dic = {}
    cache_dic['cache'] = cache
    cache_dic['num_steps'] = kwargs['num_steps']
    cache_dic['test_FLOPs'] = kwargs['test_FLOPs']
    cache_dic['monitor_gpu_usage'] = kwargs['monitor_gpu_usage']
    cache_dic['interval'] = kwargs['interval']
    cache_dic['max_order'] = kwargs['max_order']
    cache_dic['first_enhance'] = kwargs['first_enhance']

    current = {}
    current['step'] = 0
    current['activated_steps'] = [0]
    current['cache_counter'] = 0

    return cache_dic, current

