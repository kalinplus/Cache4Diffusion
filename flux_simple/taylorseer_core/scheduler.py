"""
TaylorSeer scheduler: cal_type and force_scheduler.
These two functions are tightly coupled (cal_type depends on cal_threshold written by force_scheduler).
"""
import torch


def force_scheduler(cache_dic, current):
    if cache_dic['fresh_ratio'] == 0:
        linear_step_weight = 0.0
    else:
        linear_step_weight = 0.0
    step_factor = torch.tensor(
        1 - linear_step_weight + 2 * linear_step_weight * current['step'] / current['num_steps']
    )
    threshold = torch.round(cache_dic['fresh_threshold'] / step_factor)
    cache_dic['cal_threshold'] = threshold


def cal_type(cache_dic, current):
    """Determine calculation type for this step."""
    if (cache_dic['fresh_ratio'] == 0.0) and (not cache_dic['taylor_cache']):
        first_step = (current['step'] == 0)
    else:
        first_step = (current['step'] < cache_dic['first_enhance'])

    if not first_step:
        fresh_interval = cache_dic['cal_threshold']
    else:
        fresh_interval = cache_dic['fresh_threshold']

    if (first_step) or (cache_dic['cache_counter'] == fresh_interval - 1):
        current['type'] = 'full'
        cache_dic['cache_counter'] = 0
        current['activated_steps'].append(current['step'])
        force_scheduler(cache_dic, current)

    elif cache_dic['taylor_cache']:
        cache_dic['cache_counter'] += 1
        current['type'] = 'Taylor'

    elif cache_dic['cache_counter'] % 2 == 1:
        cache_dic['cache_counter'] += 1
        current['type'] = 'ToCa'

    elif cache_dic['Delta-DiT']:
        cache_dic['cache_counter'] += 1
        current['type'] = 'Delta-Cache'

    else:
        cache_dic['cache_counter'] += 1
        current['type'] = 'ToCa'
