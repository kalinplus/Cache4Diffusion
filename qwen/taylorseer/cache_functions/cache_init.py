def cache_init(kwargs):
    '''
    Initialization for cache.
    '''
    import os

    cache = {}
    cache[-1] = {}
    cache[-2] = {}  # history for smoothing

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

    # Smoothing configuration
    cache_dic['use_smoothing'] = os.environ.get("USE_SMOOTHING", "False").lower() in ("true", "1", "yes")
    cache_dic['use_hybrid_smoothing'] = os.environ.get("USE_HYBRID_SMOOTHING", "False").lower() == "true"
    cache_dic['smoothing_method'] = os.environ.get("SMOOTHING_METHOD", "exponential")
    cache_dic['smoothing_alpha'] = float(os.environ.get("SMOOTHING_ALPHA", "0.8"))

    current = {}
    current['step'] = 0
    current['activated_steps'] = [0]
    current['cache_counter'] = 0

    return cache_dic, current

