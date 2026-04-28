import os
import torch
import math
from typing import Dict, Optional

_TS_DEBUG_SMOOTH = os.environ.get("TS_DEBUG_SMOOTH", "0").lower() in ("1", "true", "yes")
_ts_printed_steps = set()


# ─── Cache path helper ────────────────────────────────────────────────────────

def _get_module_cache(cache_dic: Dict, current: Dict, history_idx: int = -1) -> Dict:
    d = cache_dic['cache'][history_idx]
    d = d[current['module']][current['submodule']][current['subsubmodule']]
    if current['subsubmodule'] in ('resnet', 'attention') and 'idx' in current:
        d = d[current['idx']]
    if current['subsubmodule'] == 'attention' and 'subidx' in current:
        d = d[current['subidx']]
    if current['subsubmodule'] == 'attention' and 'subsubsubmodule' in current:
        d = d[current['subsubsubmodule']]
    return d


# ─── Smoothing utilities ──────────────────────────────────────────────────────

def exponential_smoothing(features, alpha: float):
    if len(features) <= 1:
        return features
    smoothed = [features[0]]
    for i in range(1, len(features)):
        smoothed.append(alpha * features[i] + (1 - alpha) * smoothed[i - 1])
    return smoothed


def moving_average_smoothing(features, window_size: int = 2):
    if len(features) < window_size:
        return features
    smoothed = []
    for i in range(len(features)):
        if i < window_size - 1:
            smoothed.append(sum(features[:i + 1]) / (i + 1))
        else:
            smoothed.append(sum(features[i - window_size + 1:i + 1]) / window_size)
    return smoothed


# ─── Original functions (unchanged signature) ────────────────────────────────

def derivative_approximation(cache_dic: Dict, current: Dict, max_order: int, first_enhance: int, feature: torch.Tensor):
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
    x = current['step'] - current['activated_steps'][-1]

    output = cache_dic[0].clone() * 0
    for i in range(len(cache_dic)):
        output += (1 / math.factorial(i)) * cache_dic[i] * (x ** i)

    return output


# ─── Unified entry point ─────────────────────────────────────────────────────

def update_cache_or_approximate(cache_dic: Dict, current: Dict, feature: Optional[torch.Tensor]):
    global _ts_printed_steps
    if current['type'] == 'full':
        max_order = cache_dic['max_order']
        first_enhance = cache_dic['first_enhance']
        difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]

        cache_now = _get_module_cache(cache_dic, current, -1)
        cache_prev = _get_module_cache(cache_dic, current, -2)

        # Save history: shallow copy cache_now → cache_prev (avoid reference aliasing)
        cache_prev.clear()
        cache_prev.update(dict(cache_now))

        # Collect raw features [F_{-2}, F_{-1}, F_0]
        raw = []
        if cache_prev.get(0) is not None:
            raw.append(cache_prev[0])
        if cache_now.get(0) is not None:
            raw.append(cache_now[0])
        raw.append(feature)

        use_smoothing = cache_dic.get('use_smoothing', False)
        use_hybrid = cache_dic.get('use_hybrid_smoothing', False)
        method = cache_dic.get('smoothing_method', 'exponential')
        alpha = cache_dic.get('smoothing_alpha', 0.8)

        updated = {}
        updated[0] = feature
        smooth_applied = False

        if use_smoothing and len(raw) >= 3 and current['step'] > first_enhance - 2:
            # Shape check
            if raw[0].shape == raw[1].shape == raw[2].shape:
                if method == 'moving_average':
                    smoothed = moving_average_smoothing(raw, window_size=2)
                else:
                    smoothed = exponential_smoothing(raw, alpha)

                smooth_applied = True

                if use_hybrid:
                    # Hybrid: 1st-order uses raw features, 2nd+ uses smoothed
                    updated[1] = (raw[-1] - raw[-2]) / difference_distance
                    if max_order >= 2 and len(smoothed) >= 3:
                        d1_now = (smoothed[-1] - smoothed[-2]) / difference_distance
                        d1_prev = (smoothed[-2] - smoothed[-3]) / difference_distance
                        updated[2] = (d1_now - d1_prev) / difference_distance
                        for i in range(2, max_order):
                            prev_factor = cache_prev.get(i, None)
                            if prev_factor is not None:
                                updated[i + 1] = (updated[i] - prev_factor) / difference_distance
                            else:
                                break
                else:
                    # Global smoothing: all derivatives from smoothed features
                    updated[0] = smoothed[-1]
                    if len(smoothed) >= 2:
                        updated[1] = (smoothed[-1] - smoothed[-2]) / difference_distance
                        for i in range(1, max_order):
                            prev_factor = cache_prev.get(i, None)
                            if prev_factor is not None:
                                updated[i + 1] = (updated[i] - prev_factor) / difference_distance
                            else:
                                break
            else:
                # Shape mismatch — fall back to non-smoothed
                for i in range(max_order):
                    if cache_now.get(i) is not None:
                        updated[i + 1] = (updated[i] - cache_now[i]) / difference_distance
                    else:
                        break
        else:
            # No smoothing — standard derivative computation
            for i in range(max_order):
                if (cache_now.get(i) is not None) and (current['step'] > first_enhance - 2):
                    updated[i + 1] = (updated[i] - cache_now[i]) / difference_distance
                else:
                    break

        cache_now.clear()
        cache_now.update(updated)

        # Debug: print once per step (first module only)
        if _TS_DEBUG_SMOOTH and current['step'] not in _ts_printed_steps:
            _ts_printed_steps.add(current['step'])
            mode = 'smooth' if smooth_applied else 'plain'
            hist = f"raw_len={len(raw)}" if use_smoothing else "no_smooth"
            print(f"[TS] step={current['step']:>2d}  type=full  {current['module']}/{current['submodule']}/{current['subsubmodule']}  "
                  f"mode={mode}  {hist}  orders={sorted(updated.keys())}  "
                  f"alpha={alpha if use_smoothing else '-'}")

        return feature
    else:
        cache_now = _get_module_cache(cache_dic, current, -1)

        if _TS_DEBUG_SMOOTH and current['step'] not in _ts_printed_steps:
            _ts_printed_steps.add(current['step'])
            print(f"[TS] step={current['step']:>2d}  type=cache  {current['module']}/{current['submodule']}/{current['subsubmodule']}  "
                  f"orders={sorted(cache_now.keys())}")

        return taylor_formula(cache_now, current)


# ─── Pipe patching ────────────────────────────────────────────────────────────

def pipe_with_cache(pipe):

    import types
    from models.unets.unet_2d_condition import UNet2DConditionModel
    from models.unets.unet_2d_blocks import CrossAttnDownBlock2D, DownBlock2D, UNetMidBlock2DCrossAttn, UpBlock2D, CrossAttnUpBlock2D
    from models.resnet import ResnetBlock2D
    from models.transformers.transformer_2d import Transformer2DModel
    from models.attention import BasicTransformerBlock

    pipe.unet.forward = types.MethodType(UNet2DConditionModel.forward, pipe.unet)
    for i, block in enumerate(pipe.unet.down_blocks):
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
