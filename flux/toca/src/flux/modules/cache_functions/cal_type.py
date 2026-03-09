def cal_type(cache_dic, current):
    '''
    Determine calculation type for this step
    '''
    first_step = (current['step'] <= cache_dic['first_enhance'] - 1)

    if (first_step) or (current['cache_counter'] == cache_dic['interval'] - 1 ):
        current['type'] = 'full'
        current['cache_counter'] = 0

    else:
        current['cache_counter'] += 1
        current['type'] = 'ToCa'
