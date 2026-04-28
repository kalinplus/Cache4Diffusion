def cache_init(num_steps, model_kwargs=None):
    '''
    Initialization for cache.
    '''
    import os

    first_enhance = int(os.environ.get('TAYLOR_FIRST_ENHANCE', 1))
    max_order = int(os.environ.get('TAYLOR_MAX_ORDER', 1))
    fresh_threshold = int(os.environ.get('TAYLOR_FRESH_THRESHOLD', 5))

    # Smooth config
    use_smoothing = os.environ.get('TAYLOR_USE_SMOOTHING', 'False').lower() in ('true', '1', 'yes')
    use_hybrid_smoothing = os.environ.get('TAYLOR_USE_HYBRID_SMOOTHING', 'False').lower() in ('true', '1', 'yes')
    smoothing_method = os.environ.get('TAYLOR_SMOOTHING_METHOD', 'exponential')
    smoothing_alpha = float(os.environ.get('TAYLOR_SMOOTHING_ALPHA', '0.8'))

    print(f"[TaylorSeer] first_enhance={first_enhance}, max_order={max_order}, fresh_threshold={fresh_threshold}")
    print(f"[TaylorSeer] use_smoothing={use_smoothing}, use_hybrid_smoothing={use_hybrid_smoothing}, "
          f"smoothing_method={smoothing_method!r}, smoothing_alpha={smoothing_alpha}")

    cache_dic = {}
    cache = {}
    cache_index = {}
    cache_index['layer_index']={}
    cache_dic['attn_map'] = {}
    cache_dic['k-norm'] = {}
    cache_dic['v-norm'] = {}
    cache_dic['cross_attn_map'] = {}
    cache_dic['cache_counter'] = 0

    for hist in [-1, -2]:
        cache[hist] = {}
        cache_index[hist] = {}
        cache_dic['attn_map'][hist] = {}
        cache_dic['k-norm'][hist] = {}
        cache_dic['v-norm'][hist] = {}

        cache[hist]['double_stream'] = {}
        cache[hist]['single_stream'] = {}
        cache_dic['attn_map'][hist]['double_stream'] = {}
        cache_dic['attn_map'][hist]['single_stream'] = {}
        cache_dic['k-norm'][hist]['double_stream'] = {}
        cache_dic['k-norm'][hist]['single_stream'] = {}
        cache_dic['v-norm'][hist]['double_stream'] = {}
        cache_dic['v-norm'][hist]['single_stream'] = {}

        for j in range(20):
            cache[hist]['double_stream'][j] = {}
            cache_index[hist][j] = {}
            cache_dic['attn_map'][hist]['double_stream'][j] = {}
            cache_dic['attn_map'][hist]['double_stream'][j]['total'] = {}
            cache_dic['attn_map'][hist]['double_stream'][j]['txt_mlp'] = {}
            cache_dic['attn_map'][hist]['double_stream'][j]['img_mlp'] = {}
            cache_dic['k-norm'][hist]['double_stream'][j] = {}
            cache_dic['k-norm'][hist]['double_stream'][j]['txt_mlp'] = {}
            cache_dic['k-norm'][hist]['double_stream'][j]['img_mlp'] = {}
            cache_dic['v-norm'][hist]['double_stream'][j] = {}
            cache_dic['v-norm'][hist]['double_stream'][j]['txt_mlp'] = {}
            cache_dic['v-norm'][hist]['double_stream'][j]['img_mlp'] = {}

        for j in range(40):
            cache[hist]['single_stream'][j] = {}
            cache_index[hist][j] = {}
            cache_dic['attn_map'][hist]['single_stream'][j] = {}
            cache_dic['attn_map'][hist]['single_stream'][j]['total'] = {}
            cache_dic['k-norm'][hist]['single_stream'][j] = {}
            cache_dic['k-norm'][hist]['single_stream'][j]['total'] = {}
            cache_dic['v-norm'][hist]['single_stream'][j] = {}
            cache_dic['v-norm'][hist]['single_stream'][j]['total'] = {}

    cache_dic['cross_attn_map'][-1] = {}

    cache_dic['taylor_cache'] = False
    cache_dic['duca']  = False
    cache_dic['test_FLOPs'] = False
    cache_dic['use_smoothing'] = use_smoothing
    cache_dic['use_hybrid_smoothing'] = use_hybrid_smoothing
    cache_dic['smoothing_method'] = smoothing_method
    cache_dic['smoothing_alpha'] = smoothing_alpha

    mode = 'Taylor'
    if mode == 'original':
        cache_dic['cache_type'] = 'random'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa'
        cache_dic['fresh_ratio'] = 0.0
        cache_dic['fresh_threshold'] = 1
        cache_dic['force_fresh'] = 'global' 
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = first_enhance

    elif mode == 'ToCa':
        cache_dic['cache_type'] = 'random'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa'
        cache_dic['fresh_ratio'] = 0.10
        cache_dic['fresh_threshold'] = fresh_threshold
        cache_dic['force_fresh'] = 'global'
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = first_enhance
        cache_dic['duca'] = False

    elif mode == 'DuCa':
        cache_dic['cache_type'] = 'random'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa'
        cache_dic['fresh_ratio'] = 0.10
        cache_dic['fresh_threshold'] = fresh_threshold
        cache_dic['force_fresh'] = 'global'
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = first_enhance
        cache_dic['duca'] = True

    elif mode == 'Taylor':
        cache_dic['cache_type'] = 'random'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa'
        cache_dic['fresh_ratio'] = 0.0
        cache_dic['fresh_threshold'] = fresh_threshold
        cache_dic['max_order'] = max_order
        cache_dic['force_fresh'] = 'global'
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['taylor_cache'] = True
        cache_dic['first_enhance'] = first_enhance

    current = {}
    current['num_steps'] = num_steps
    current['activated_steps'] = [0]

    return cache_dic, current
