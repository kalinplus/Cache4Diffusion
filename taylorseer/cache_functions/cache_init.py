def cache_init(**kwargs):   
    '''
    Initialization for cache.
    '''
    cache = {}
    cache[-1]={}

    cache[-1]['downblocks'] = {}
    for submodule in ['DownBlock2D_0', 'CrossAttnDownBlock2D_1', 'CrossAttnDownBlock2D_2']:
        cache[-1]['downblocks'][submodule] = {}
        for subsubmodule in ['resnet', 'attention', 'downsampler']:
            cache[-1]['downblocks'][submodule][subsubmodule] = {}
            if subsubmodule == 'resnet' or subsubmodule == 'attention':
                for i in range(2):
                    cache[-1]['downblocks'][submodule][subsubmodule][i] = {}
                    if submodule in ['CrossAttnDownBlock2D_1', 'CrossAttnDownBlock2D_2'] and subsubmodule == 'attention':
                        num_of_subattentions = 2 if submodule == 'CrossAttnDownBlock2D_1' else 10
                        for j in range(num_of_subattentions):
                            cache[-1]['downblocks'][submodule][subsubmodule][i][j] = {} # subattention
                            for subsubsubmodule in ['attn1', 'attn2', 'mlp']:
                                cache[-1]['downblocks'][submodule][subsubmodule][i][j][subsubsubmodule] = {}
                            
    cache[-1]['midblock'] = {}
    cache[-1]['midblock']['UNetMidBlock2DCrossAttn'] = {}
    for subsubmodule in ['resnet', 'attention']:
        cache[-1]['midblock']['UNetMidBlock2DCrossAttn'][subsubmodule] = {}
        if subsubmodule == 'resnet':
            for i in range(2):
                cache[-1]['midblock']['UNetMidBlock2DCrossAttn'][subsubmodule][i] = {}
        elif subsubmodule == 'attention':
            cache[-1]['midblock']['UNetMidBlock2DCrossAttn'][subsubmodule][0] = {}
            num_of_subattentions = 10
            for i in range(num_of_subattentions):
                cache[-1]['midblock']['UNetMidBlock2DCrossAttn'][subsubmodule][0][i] = {} # subattention
                for subsubsubmodule in ['attn1', 'attn2', 'mlp']:
                    cache[-1]['midblock']['UNetMidBlock2DCrossAttn'][subsubmodule][0][i][subsubsubmodule] = {}

    cache[-1]['upblocks'] = {}
    for submodule in ['CrossAttnUpBlock2D_0', 'CrossAttnUpBlock2D_1', 'UpBlock2D_2']:
        cache[-1]['upblocks'][submodule] = {}
        for subsubmodule in ['resnet', 'attention', 'upsampler']:
            cache[-1]['upblocks'][submodule][subsubmodule] = {}
            if subsubmodule == 'resnet' or subsubmodule == 'attention':
                for i in range(3):
                    cache[-1]['upblocks'][submodule][subsubmodule][i] = {}
                    if submodule in ['CrossAttnUpBlock2D_0', 'CrossAttnUpBlock2D_1'] and subsubmodule == 'attention':
                        num_of_subattentions = 10 if submodule == 'CrossAttnUpBlock2D_0' else 2
                        for j in range(num_of_subattentions):
                            cache[-1]['upblocks'][submodule][subsubmodule][i][j] = {} # subattention
                            for subsubsubmodule in ['attn1', 'attn2', 'mlp']:
                                cache[-1]['upblocks'][submodule][subsubmodule][i][j][subsubsubmodule] = {}
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

    current = {}
    current['step'] = 0
    current['activated_steps'] = [0]
    current['cache_counter'] = 0

    return cache_dic, current

# num_of_downblocks: 3
# num_of_mid_block: 1
# num_of_upblocks: 3
# DownBlock2D
# num_of_resnets: 2
# CrossAttnDownBlock2D
# num_of_resnets: 2
# num_of_attentions: 2
# Transformer2DModel
# num_of_subattentions: 2
# Transformer2DModel
# num_of_subattentions: 2
# CrossAttnDownBlock2D
# num_of_resnets: 2
# num_of_attentions: 2
# Transformer2DModel
# num_of_subattentions: 10
# Transformer2DModel
# num_of_subattentions: 10
# UNetMidBlock2DCrossAttn
# num_of_resnets: 2
# num_of_attentions: 1
# Transformer2DModel
# num_of_subattentions: 10
# CrossAttnUpBlock2D
# num_of_resnets: 3
# num_of_attentions: 3
# Transformer2DModel
# num_of_subattentions: 10
# Transformer2DModel
# num_of_subattentions: 10
# Transformer2DModel
# num_of_subattentions: 10
# CrossAttnUpBlock2D
# num_of_resnets: 3
# num_of_attentions: 3
# Transformer2DModel
# num_of_subattentions: 2
# Transformer2DModel
# num_of_subattentions: 2
# Transformer2DModel
# num_of_subattentions: 2
# UpBlock2D
# num_of_resnets: 3
