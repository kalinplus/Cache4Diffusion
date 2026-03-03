import torch
import math
from typing import Dict, List


def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic['max_order']):
        if (cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].get(i, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
            updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i]) / difference_distance
        else:
            break
    
    cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = updated_taylor_factors


def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor: 
    """
    Compute Taylor expansion error.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    x = current['step'] - current['activated_steps'][-1]

    output = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][0].clone() * 0

    for i in range(len(cache_dic['cache'][-1][current['stream']][current['layer']][current['module']])):
        output += (1 / math.factorial(i)) * cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i] * (x ** i)
    
    return output


def module_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache and allocate storage for different-order derivatives in the Taylor cache.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if (current['step'] == 0):
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = {}


def pipeline_with_cache(pipe):

    import types
    from pipeline.pipeline_qwenimage import QwenImagePipeline as LocalQwenImagePipeline
    from pipeline.transformer_qwenimage import QwenImageTransformer2DModel as LocalQwenImageTransformer2DModel
    from pipeline.transformer_qwenimage import QwenImageTransformerBlock as LocalQwenImageTransformerBlock

    pipe.__call__ = types.MethodType(LocalQwenImagePipeline.__call__, pipe)

    pipe.transformer.forward = types.MethodType(
        LocalQwenImageTransformer2DModel.forward, pipe.transformer
    )

    for _, block in enumerate(pipe.transformer.transformer_blocks):
        block.forward = types.MethodType(LocalQwenImageTransformerBlock.forward, block)

    return pipe





def exponential_smoothing(features: List[torch.Tensor], alpha: float) -> List[torch.Tensor]:
    """
    指数平滑滤波：对历史特征序列进行平滑处理。
    :param features: 特征列表 [x1, x2, x3, ...]，每个是 (N, D) 的 tensor
    :param alpha: 平滑系数，0 < alpha < 1，越小平滑越强
    :return: 平滑后的特征列表
    """
    if len(features) <= 1:
        return features

    smoothed = [features[0]]
    for i in range(1, len(features)):
        # S_t = alpha * X_t + (1 - alpha) * S_{t-1}
        smoothed_val = alpha * features[i] + (1 - alpha) * smoothed[i - 1]
        smoothed.append(smoothed_val)

    return smoothed


def moving_average_smoothing(features: List[torch.Tensor], window_size: int = 2) -> List[torch.Tensor]:
    """
    移动平均滤波：对历史特征序列进行平滑处理。
    :param features: 特征列表 [x1, x2, x3, ...]，每个是 (N, D) 的 tensor
    :param window_size: 窗口大小
    :return: 平滑后的特征列表
    """
    if len(features) < window_size:
        return features

    smoothed = []
    for i in range(len(features)):
        if i < window_size - 1:
            smoothed.append(sum(features[:i+1]) / (i + 1))
        else:
            # 计算窗口内的平均值
            window_sum = sum(features[i - window_size + 1 : i + 1])
            smoothed.append(window_sum / window_size)

    return smoothed


def shift_cache_history(cache_dic: Dict, current: Dict):
    """
    Shift cache[-1] -> cache[-2] for smoothing.
    Adapted for cache_utils.py structure.
    """
    cache = cache_dic["cache"]
    s, l, m = current["stream"], current["layer"], current["module"]

    if current["step"] == 0:
        cache[-2][s][l][m] = {}
        return

    cache[-2][s][l][m] = cache[-1][s][l][m]
    # print(f"[TS-SMOOTH] Step {current['step']}: Shifted cache[-1] -> cache[-2] for {s}/{l}/{m}")


def _collect_history_f0_fm1_fm2(cache_dic: Dict, current: Dict, feature: torch.Tensor) -> List[torch.Tensor]:
    """
    Collect raw features: [F_{-2}, F_{-1}, F_0].
    Adapted for cache_utils.py structure.
    """
    cache = cache_dic["cache"]
    s, l, m = current["stream"], current["layer"], current["module"]
    entry_m1 = cache[-1][s][l][m]
    entry_m2 = cache[-2][s][l][m]

    feats = []
    if entry_m2.get(0, None) is not None:
        feats.append(entry_m2[0])
    if entry_m1.get(0, None) is not None:
        feats.append(entry_m1[0])
    feats.append(feature)

    return feats


def derivative_approximation_with_smoothing(
    cache_dic: Dict,
    current: Dict,
    feature: torch.Tensor,
    smoothing_method: str = "exponential",
    alpha: float = 0.7,
    window_size: int = 2,
):
    """
    Derivative approximation with smoothing.
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    :param feature: Current feature tensor
    :param smoothing_method: Smoothing method ("exponential" or "moving_average")
    :param alpha: Alpha parameter for exponential smoothing
    :param window_size: Window size for moving average smoothing
    """
    s, l, m = current["stream"], current["layer"], current["module"]
    # print(f"[TS-SMOOTH] Step {current['step']}: derivative_with_smoothing for {s}/{l}/{m}, method={smoothing_method}, alpha={alpha}")

    difference_distance = current["activated_steps"][-1] - current["activated_steps"][-2]

    # Collect history features
    raw_features = _collect_history_f0_fm1_fm2(cache_dic, current, feature)

    cache = cache_dic["cache"]
    s, l, m = current["stream"], current["layer"], current["module"]
    cache_entry = cache[-1][s][l][m]

    updated_taylor_factors = {}

    if len(raw_features) >= 3:
        # Apply smoothing
        if smoothing_method == "moving_average":
            smoothed = moving_average_smoothing(raw_features, window_size=window_size)
        else:
            smoothed = exponential_smoothing(raw_features, alpha=alpha)

        updated_taylor_factors[0] = smoothed[-1]
        updated_taylor_factors[1] = (smoothed[-1] - smoothed[-2]) / difference_distance
    elif len(raw_features) == 2:
        updated_taylor_factors[0] = feature
        updated_taylor_factors[1] = (raw_features[-1] - raw_features[-2]) / difference_distance
    else:
        updated_taylor_factors[0] = feature

    # Compute higher order derivatives
    for i in range(1, cache_dic['max_order']):
        if (i in updated_taylor_factors) and (cache_entry.get(i, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
            updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_entry[i]) / difference_distance
        else:
            break

    cache[-1][s][l][m] = updated_taylor_factors