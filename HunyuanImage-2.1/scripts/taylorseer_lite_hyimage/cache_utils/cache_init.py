
def cache_init(num_steps, use_smoothing=False, smoothing_method='exponential', smoothing_alpha=0.8):
    '''
    Initialization for cache.

    Cache parameters can be set via environment variables:
    - TS_CACHE_INTERVAL: Cache interval (default: 5)
    - TS_MAX_ORDER: Maximum Taylor order (default: 2)
    - TS_FIRST_ENHANCE: First enhancement step (default: 3)
    '''
    import os

    cache_dic = {}
    cache = {}
    cache_dic['cache_counter'] = 0
    for i in [-1, -2]:
        cache[i]={}

        cache[i]['double_stream']={}
        cache[i]['single_stream']={}

        cache[i]['final'] = {}
        cache[i]['final']['final'] = {}
        cache[i]['final']['final']['final'] = {}

    cache_dic['cache'] = cache

    # Read cache configuration from environment variables with defaults
    cache_dic['cache_interval'] = int(os.environ.get('TS_CACHE_INTERVAL', 5))
    cache_dic['max_order'] = int(os.environ.get('TS_MAX_ORDER', 2))
    cache_dic['first_enhance'] = int(os.environ.get('TS_FIRST_ENHANCE', 3))

    cache_dic['taylor_cache'] = True

    # Smoothing configuration
    cache_dic['use_smoothing'] = use_smoothing
    cache_dic['smoothing_method'] = smoothing_method
    cache_dic['smoothing_alpha'] = smoothing_alpha

    current = {}
    current['activated_steps'] = [0]

    current['step'] = 0
    current['num_steps'] = num_steps

    return cache_dic, current
