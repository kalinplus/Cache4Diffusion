import copy


def _create_block_cache():
    cache = {}
    cache['downblocks'] = {}
    for submodule in ['DownBlock2D_0', 'CrossAttnDownBlock2D_1', 'CrossAttnDownBlock2D_2']:
        cache['downblocks'][submodule] = {}
        for subsubmodule in ['resnet', 'attention', 'downsampler']:
            cache['downblocks'][submodule][subsubmodule] = {}
            if subsubmodule == 'resnet' or subsubmodule == 'attention':
                for i in range(2):
                    cache['downblocks'][submodule][subsubmodule][i] = {}
                    if submodule in ['CrossAttnDownBlock2D_1', 'CrossAttnDownBlock2D_2'] and subsubmodule == 'attention':
                        num_of_subattentions = 2 if submodule == 'CrossAttnDownBlock2D_1' else 10
                        for j in range(num_of_subattentions):
                            cache['downblocks'][submodule][subsubmodule][i][j] = {}
                            for subsubsubmodule in ['attn1', 'attn2', 'mlp']:
                                cache['downblocks'][submodule][subsubmodule][i][j][subsubsubmodule] = {}

    cache['midblock'] = {}
    cache['midblock']['UNetMidBlock2DCrossAttn'] = {}
    for subsubmodule in ['resnet', 'attention']:
        cache['midblock']['UNetMidBlock2DCrossAttn'][subsubmodule] = {}
        if subsubmodule == 'resnet':
            for i in range(2):
                cache['midblock']['UNetMidBlock2DCrossAttn'][subsubmodule][i] = {}
        elif subsubmodule == 'attention':
            cache['midblock']['UNetMidBlock2DCrossAttn'][subsubmodule][0] = {}
            num_of_subattentions = 10
            for i in range(num_of_subattentions):
                cache['midblock']['UNetMidBlock2DCrossAttn'][subsubmodule][0][i] = {}
                for subsubsubmodule in ['attn1', 'attn2', 'mlp']:
                    cache['midblock']['UNetMidBlock2DCrossAttn'][subsubmodule][0][i][subsubsubmodule] = {}

    cache['upblocks'] = {}
    for submodule in ['CrossAttnUpBlock2D_0', 'CrossAttnUpBlock2D_1', 'UpBlock2D_2']:
        cache['upblocks'][submodule] = {}
        for subsubmodule in ['resnet', 'attention', 'upsampler']:
            cache['upblocks'][submodule][subsubmodule] = {}
            if subsubmodule == 'resnet' or subsubmodule == 'attention':
                for i in range(3):
                    cache['upblocks'][submodule][subsubmodule][i] = {}
                    if submodule in ['CrossAttnUpBlock2D_0', 'CrossAttnUpBlock2D_1'] and subsubmodule == 'attention':
                        num_of_subattentions = 10 if submodule == 'CrossAttnUpBlock2D_0' else 2
                        for j in range(num_of_subattentions):
                            cache['upblocks'][submodule][subsubmodule][i][j] = {}
                            for subsubsubmodule in ['attn1', 'attn2', 'mlp']:
                                cache['upblocks'][submodule][subsubmodule][i][j][subsubsubmodule] = {}

    return cache


def cache_init(**kwargs):
    cache = {}
    cache[-1] = _create_block_cache()
    cache[-2] = _create_block_cache()

    cache_dic = {}
    cache_dic['cache'] = cache
    cache_dic['height'] = kwargs['height']
    cache_dic['width'] = kwargs['width']
    cache_dic['num_steps'] = kwargs['num_steps']
    cache_dic['test_FLOPs'] = kwargs['test_FLOPs']
    cache_dic['monitor_gpu_usage'] = kwargs['monitor_gpu_usage']
    cache_dic['interval'] = kwargs['interval']
    cache_dic['max_order'] = kwargs['max_order']
    cache_dic['first_enhance'] = kwargs['first_enhance']

    cache_dic['use_smoothing'] = kwargs.get('use_smoothing', False)
    cache_dic['use_hybrid_smoothing'] = kwargs.get('use_hybrid_smoothing', False)
    cache_dic['smoothing_method'] = kwargs.get('smoothing_method', 'exponential')
    cache_dic['smoothing_alpha'] = kwargs.get('smoothing_alpha', 0.8)

    current = {}
    current['step'] = 0
    current['activated_steps'] = [0]
    current['cache_counter'] = 0

    return cache_dic, current
