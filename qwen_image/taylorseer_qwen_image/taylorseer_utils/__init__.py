import os
import json
from typing import Dict 
import torch
import math
from collections import Counter

def derivative_approximation(cache_dic, current, feature):
    """
    Compute derivative approximation.
    """
    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    cache_entry = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]

    for i in range(cache_dic['max_order']):
        if cache_entry.get(i, None) is not None:
            updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_entry[i]) / difference_distance
        else:
            break

    cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = updated_taylor_factors

def taylor_formula(cache_dic, current):
    """
    Compute Taylor expansion.
    """
    x = current['step'] - current['activated_steps'][-1]

    cache_entry = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]

    output = 0
    for i in range(len(cache_entry)):
        output += (1 / math.factorial(i)) * cache_entry[i] * (x ** i)

    return output

def module_cache_init(cache_dic, current):
    """
    Initialize Taylor cache.
    """
    if (current['step'] == 0) and (cache_dic['taylor_cache']):
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = {}



_TS_DEBUG_SHAPES = os.environ.get("TS_DEBUG_SHAPES", "0").lower() in ("1", "true", "yes")
_TS_STRICT_SHAPES = os.environ.get("TS_STRICT_SHAPES", "0").lower() in ("1", "true", "yes")

def _shape_str(x):
    try:
        return str(tuple(x.shape))
    except Exception:
        return str(type(x))

def _debug_prefix(cache_dic, current):
    return (
        f"[TS][step={current.get('step')} act={current.get('activated_steps')}]"
        f"[branch={current.get('branch','?')}]"
        f"[stream={current.get('stream','?')}, layer={current.get('layer','?')}, module={current.get('module','?')}]"
    )

def exponential_smoothing(features: list, alpha: float) -> list:
    """
    指数平滑滤波：对历史特征序列进行平滑处理。
    
    :param features: 特征列表 [x1, x2, x3, ...]，每个是 (N, D) 的 tensor
    :param alpha: 平滑系数，0 < alpha < 1，越小平滑越强
    :return: 平滑后的特征列表
    """
    if len(features) <= 1:
        return features
    
    smoothed = [features[0]]  # 第一个点保持不变
    for i in range(1, len(features)):
        # S_t = alpha * X_t + (1 - alpha) * S_{t-1}
        smoothed_val = alpha * features[i] + (1 - alpha) * smoothed[i - 1]
        smoothed.append(smoothed_val)
    
    return smoothed

def moving_average_smoothing(features: list, window_size: int = 2) -> list:
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
            # 窗口不足时，使用可用的点
            smoothed.append(sum(features[:i+1]) / (i + 1))
        else:
            # 计算窗口内的平均值
            window_sum = sum(features[i - window_size + 1 : i + 1])
            smoothed.append(window_sum / window_size)
    
    return smoothed

def shift_cache_history(cache_dic, current):
    """
    Shift cache[-1] -> cache[-2] for smoothing.
    """
    if not cache_dic.get("taylor_cache", False):
        return

    cache = cache_dic["cache"]
    s, l, m = current["stream"], current["layer"], current["module"]

    if current["step"] == 0:
        cache[-2][s][l][m] = {}
        return

    cache[-2][s][l][m] = cache[-1][s][l][m]

def _collect_history_f0_fm1_fm2(cache_dic, current, feature):
    """
    Collect raw features: [F_{-2}, F_{-1}, F_0].
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
    cache_dic,
    current,
    feature,
    smoothing_method="exponential",
    alpha=0.7,
    window_size=2,
):
    h = current["activated_steps"][-1] - current["activated_steps"][-2]

    # NEW: print context once in a while if enabled
    raw = _collect_history_f0_fm1_fm2(cache_dic, current, feature)

    if _TS_DEBUG_SHAPES:
        shapes = ", ".join(_shape_str(t) for t in raw)
        print(_debug_prefix(cache_dic, current), f"raw_len={len(raw)} raw_shapes=[{shapes}] alpha={alpha} method={smoothing_method}")

    # NEW: guard against mismatched shapes (your current crash)
    if len(raw) >= 2:
        base = tuple(raw[-1].shape)
        mismatch = any(tuple(t.shape) != base for t in raw[:-1])
        if mismatch:
            msg = _debug_prefix(cache_dic, current) + " SHAPE_MISMATCH raw_shapes=[" + ", ".join(_shape_str(t) for t in raw) + "]"
            if _TS_STRICT_SHAPES:
                raise RuntimeError(msg)
            # fallback: disable smoothing for this update (still update cache so pipeline continues)
            if _TS_DEBUG_SHAPES:
                print(msg + " -> fallback to non-smoothing update")
            return derivative_approximation(cache_dic, current, feature)

    cache = cache_dic["cache"]
    s, l, m = current["stream"], current["layer"], current["module"]
    cache_entry = cache[-1][s][l][m]

    updated = {}
    if len(raw) >= 3:
        if smoothing_method == "moving_average":
            smoothed = moving_average_smoothing(raw, window_size=window_size)
        else:
            smoothed = exponential_smoothing(raw, alpha=alpha)
        updated[0] = smoothed[-1]
        updated[1] = (smoothed[-1] - smoothed[-2]) / h
    elif len(raw) == 2:
        updated[0] = feature
        updated[1] = (raw[-1] - raw[-2]) / h
    else:
        updated[0] = feature

    for order in range(1, cache_dic["max_order"]):
        if (order in updated) and (cache_entry.get(order, None) is not None):
            updated[order + 1] = (updated[order] - cache_entry[order]) / h
        else:
            break

    cache[-1][s][l][m] = updated

def derivative_approximation_hybrid_smoothing(
    cache_dic,
    current,
    feature,
    smoothing_method="exponential",
    alpha=0.7,
    window_size=2,
):
    """
    Hybrid smoothing.
    """
    h = current["activated_steps"][-1] - current["activated_steps"][-2]
    cache = cache_dic["cache"]
    s, l, m = current["stream"], current["layer"], current["module"]
    cache_entry = cache[-1][s][l][m]
    
    raw = _collect_history_f0_fm1_fm2(cache_dic, current, feature)
    updated = {0: feature}
    
    if len(raw) >= 2:
        updated[1] = (raw[-1] - raw[-2]) / h
    
    if len(raw) >= 3 and cache_dic["max_order"] >= 2:
        if smoothing_method == "moving_average":
            smoothed = moving_average_smoothing(raw, window_size=window_size)
        else:
            smoothed = exponential_smoothing(raw, alpha=alpha)
        
        d1_now = (smoothed[-1] - smoothed[-2]) / h
        d1_prev = (smoothed[-2] - smoothed[-3]) / h
        updated[2] = (d1_now - d1_prev) / h
        
        for order in range(2, cache_dic["max_order"]):
            if (order in updated) and (cache_entry.get(order, None) is not None):
                updated[order + 1] = (updated[order] - cache_entry[order]) / h
            else:
                break
    else:
        for order in range(1, cache_dic["max_order"]):
            if (order in updated) and (cache_entry.get(order, None) is not None):
                updated[order + 1] = (updated[order] - cache_entry[order]) / h
            else:
                break
    
    cache[-1][s][l][m] = updated
