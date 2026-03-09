def cal_type(cache_dic, current):
    '''
    Determine calculation type for this step
    '''
    first_step = (current['step'] <= cache_dic['first_enhance'] - 1)

    if (first_step) or (current['cache_counter'] == cache_dic['interval'] - 1 ):
        current['type'] = 'full'
        current['cache_counter'] = 0

    elif (current['cache_counter'] % 2 == 0): # 0: DuCa-Aggresive-DuCa, 1: Aggresive-DuCa-Aggresive
        current['cache_counter'] += 1
        current['type'] = 'DuCa'

    else:
        current['cache_counter'] += 1 
        if current['step'] < 25:
            current['type'] = 'FORA'
        else:
            current['type'] = 'aggressive'
