from diffusers.models import QwenImageTransformer2DModel
import os

N = 6
O = 1

USE_SMOOTHING = os.environ.get("USE_SMOOTHING", "False").lower() in ("true", "1", "yes")  # 是否启用平滑
USE_HYBRID_SMOOTHING = os.environ.get("USE_HYBRID_SMOOTHING", "False").lower() == "true" # 仅当 USE_SMOOTHING=True 时有效
SMOOTHING_METHOD = os.environ.get("SMOOTHING_METHOD", "exponential")  # 'exponential' or 'moving_average'
SMOOTHING_ALPHA = float(os.environ.get("SMOOTHING_ALPHA", "0.7"))  # for exponential smoothing


def cache_init(self: QwenImageTransformer2DModel):   
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache_index = {}
    for history_index in [-2, -1]:
        cache[history_index]={}
        cache_index[history_index]={}
        cache[history_index]['double_stream']={}

    cache_index['layer_index']={}
    cache_dic['attn_map'] = {}
    cache_dic['attn_map'][-1] = {}
    # cache_dic['attn_map'][-1]['double_stream'] = {}
    # cache_dic['attn_map'][-1]['single_stream'] = {}
    cache_dic['attn_map'][-1]['double_stream'] = {}

    # cache[-1]['double_stream']={}
    # cache[-1]['single_stream']={}
    cache_dic['cache_counter'] = 0

    # for j in range(60):
    for i in range(self.config.num_layers):
        for history_index in [-2, -1]:
            cache[history_index]['double_stream'][i] = {}
        cache_index[-1][i] = {}
        cache_dic['attn_map'][-1]['double_stream'][i] = {}
        cache_dic['attn_map'][-1]['double_stream'][i]['total'] = {}

    # #for j in range(19):
    # for j in range(self.config.num_layers):
    #     cache[-1]['double_stream'][j] = {}
    #     cache_index[-1][j] = {}
    #     cache_dic['attn_map'][-1]['double_stream'][j] = {}
    #     cache_dic['attn_map'][-1]['double_stream'][j]['total'] = {}
    #     cache_dic['attn_map'][-1]['double_stream'][j]['txt_mlp'] = {}
    #     cache_dic['attn_map'][-1]['double_stream'][j]['img_mlp'] = {}

    # #for j in range(38):
    # for j in range(self.config.num_single_layers):
    #     cache[-1]['single_stream'][j] = {}
    #     cache_index[-1][j] = {}
    #     cache_dic['attn_map'][-1]['single_stream'][j] = {}
    #     cache_dic['attn_map'][-1]['single_stream'][j]['total'] = {}

    cache_dic['taylor_cache'] = False
    cache_dic['Delta-DiT'] = False

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
        cache_dic['first_enhance'] = 3
    
    elif mode == 'Taylor':
        cache_dic['cache_type'] = 'random'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa' 
        cache_dic['fresh_ratio'] = 0.0
        cache_dic['fresh_threshold'] = N
        cache_dic['force_fresh'] = 'global' 
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['taylor_cache'] = True
        cache_dic['max_order'] = O
        cache_dic['first_enhance'] = 3  # there is first enhance, no problem

    # 新增：平滑相关配置
    cache_dic['use_smoothing'] = USE_SMOOTHING  # 是否启用平滑
    cache_dic['use_hybrid_smoothing'] = USE_HYBRID_SMOOTHING  # 是否启用混合平滑, 仅当 use_smoothing=True 时有效
    cache_dic['smoothing_method'] = SMOOTHING_METHOD # 'exponential' 或 'moving_average'
    cache_dic['smoothing_alpha'] = SMOOTHING_ALPHA  # 指数平滑系数
    cache_dic['smoothed_derivatives'] = {}  # 用于混合平滑

    current = {}
    current['activated_steps'] = [0]
    current['step'] = 0
    current['num_steps'] = self.num_steps

    return cache_dic, current
