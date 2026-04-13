import torch
import math
from typing import Dict


def derivative_approximation(cache_dic: Dict, current: Dict, max_order: int, first_enhance: int, feature: torch.Tensor):
    """
    Compute derivative approximation.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(max_order):
        if (cache_dic.get(i, None) is not None) and (current['step'] > first_enhance - 2):
            updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_dic[i]) / difference_distance
        else:
            break
    
    return updated_taylor_factors


def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor: 
    """
    Compute Taylor expansion error.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    x = current['step'] - current['activated_steps'][-1]

    output = cache_dic[0].clone() * 0
    for i in range(len(cache_dic)):
        output += (1 / math.factorial(i)) * cache_dic[i] * (x ** i)
    
    return output


def pipe_with_cache(pipe):

    import types
    from models.unets.unet_2d_condition import UNet2DConditionModel
    from models.unets.unet_2d_blocks import CrossAttnDownBlock2D, DownBlock2D, UNetMidBlock2DCrossAttn, UpBlock2D, CrossAttnUpBlock2D
    from models.resnet import ResnetBlock2D
    from models.transformers.transformer_2d import Transformer2DModel
    from models.attention import BasicTransformerBlock

    pipe.unet.forward = types.MethodType(UNet2DConditionModel.forward, pipe.unet)
    for i, block in enumerate(pipe.unet.down_blocks):
        # print(i, block.__class__.__name__)
        if i == 0:
            block.forward = types.MethodType(DownBlock2D.forward, block)
            for _, resnet in enumerate(block.resnets):
                resnet.forward = types.MethodType(ResnetBlock2D.forward, resnet)
        else:
            block.forward = types.MethodType(CrossAttnDownBlock2D.forward, block)
            for _, resnet in enumerate(block.resnets):
                resnet.forward = types.MethodType(ResnetBlock2D.forward, resnet)
            for _, attention in enumerate(block.attentions):
                attention.forward = types.MethodType(Transformer2DModel.forward, attention)
                for _, subattention in enumerate(attention.transformer_blocks):
                    subattention.forward = types.MethodType(BasicTransformerBlock.forward, subattention)
    
    pipe.unet.mid_block.forward = types.MethodType(UNetMidBlock2DCrossAttn.forward, pipe.unet.mid_block)
    for _, resnet in enumerate(pipe.unet.mid_block.resnets):
        resnet.forward = types.MethodType(ResnetBlock2D.forward, resnet)
    for _, attention in enumerate(pipe.unet.mid_block.attentions):
        attention.forward = types.MethodType(Transformer2DModel.forward, attention)
        for _, subattention in enumerate(attention.transformer_blocks):
            subattention.forward = types.MethodType(BasicTransformerBlock.forward, subattention)

    for i, block in enumerate(pipe.unet.up_blocks):
        # print(i, block.__class__.__name__)
        if i == 2:
            block.forward = types.MethodType(UpBlock2D.forward, block)
            for _, resnet in enumerate(block.resnets):
                resnet.forward = types.MethodType(ResnetBlock2D.forward, resnet)
        else:
            block.forward = types.MethodType(CrossAttnUpBlock2D.forward, block)
            for _, resnet in enumerate(block.resnets):
                resnet.forward = types.MethodType(ResnetBlock2D.forward, resnet)
            for _, attention in enumerate(block.attentions):
                attention.forward = types.MethodType(Transformer2DModel.forward, attention)
                for _, subattention in enumerate(attention.transformer_blocks):
                    subattention.forward = types.MethodType(BasicTransformerBlock.forward, subattention)

    return pipe