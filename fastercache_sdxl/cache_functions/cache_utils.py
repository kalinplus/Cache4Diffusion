import torch
from typing import Dict


def cache_store(cache_entry: Dict, feature: torch.Tensor):
    """
    Store feature after a full step: old <- new, new <- feature.
    """
    cache_entry["old"] = cache_entry.get("new")
    cache_entry["new"] = feature


def cache_predict(cache_entry: Dict, alpha: float) -> torch.Tensor:
    """
    Predict feature for a cache step via linear extrapolation.
    """
    if cache_entry.get("old") is not None:
        return cache_entry["new"] + alpha * (cache_entry["new"] - cache_entry["old"])
    else:
        return cache_entry["new"]


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