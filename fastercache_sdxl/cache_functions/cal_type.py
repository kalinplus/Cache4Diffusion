def cal_type(cache_dic, current):
    '''
    Determine calculation type for this step.
    Aligned with taylorseer logic: warm-up steps + periodic full steps via cache_counter.
    '''
    first_step = (current['step'] < cache_dic['fc_start_step'])

    if (first_step) or (current['cache_counter'] == cache_dic['fc_interval'] - 1):
        current['type'] = 'full'
        current['cache_counter'] = 0
    else:
        current['type'] = 'cache'
        current['cache_counter'] += 1
