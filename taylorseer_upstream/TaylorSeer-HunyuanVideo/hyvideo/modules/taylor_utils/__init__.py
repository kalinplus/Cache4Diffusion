from typing import Dict
import torch
import math
from .convert_flops import convert_flops


# ---------------------------------------------------------------------------
# Core Taylor functions (original)
# ---------------------------------------------------------------------------

def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]
    #difference_distance = current['activated_times'][-1] - current['activated_times'][-2]

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
    Compute Taylor expansion
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    x = current['step'] - current['activated_steps'][-1]
    #x = current['t'] - current['activated_times'][-1]
    output = 0

    for i in range(len(cache_dic['cache'][-1][current['stream']][current['layer']][current['module']])):
        output += (1 / math.factorial(i)) * cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i] * (x ** i)

    return output

def taylor_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache, expanding storage areas for Taylor series derivatives
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if current['step'] == 0:
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = {}


# ---------------------------------------------------------------------------
# Smooth algorithm functions
# ---------------------------------------------------------------------------

def shift_cache_history(cache_dic: Dict, current: Dict):
    """
    Shift cache[-1] to cache[-2] before a new full compute step.
    """
    if not cache_dic.get("taylor_cache", False):
        return

    cache = cache_dic["cache"]
    s, l, m = current["stream"], current["layer"], current["module"]

    if current["step"] == 0:
        cache[-2][s][l][m] = {}
        return

    cache[-2][s][l][m] = cache[-1][s][l][m]


def exponential_smoothing(features: list, alpha: float) -> list:
    """Apply exponential smoothing to a list of feature tensors."""
    if len(features) <= 1:
        return features
    smoothed = [features[0]]
    for i in range(1, len(features)):
        smoothed.append(alpha * features[i] + (1 - alpha) * smoothed[i - 1])
    return smoothed


def moving_average_smoothing(features: list, window_size: int = 2) -> list:
    """Apply moving average smoothing to a list of feature tensors."""
    if len(features) < window_size:
        return features
    smoothed = []
    for i in range(len(features)):
        if i < window_size - 1:
            smoothed.append(sum(features[: i + 1]) / (i + 1))
        else:
            smoothed.append(sum(features[i - window_size + 1 : i + 1]) / window_size)
    return smoothed


def derivative_approximation_with_smoothing(cache_dic: Dict, current: Dict, feature: torch.Tensor,
                                            method: str = 'exponential', alpha: float = 0.8):
    """
    Compute derivative approximation with global smoothing.
    All derivatives are computed from smoothed features.
    """
    s, l, m = current["stream"], current["layer"], current["module"]
    h = current['activated_steps'][-1] - current['activated_steps'][-2]

    # Collect historical features: [F_{-2}, F_{-1}, F_0]
    f_prev2 = cache_dic["cache"][-2][s][l][m].get(0, None)
    f_prev1 = cache_dic["cache"][-1][s][l][m].get(0, None)

    if f_prev2 is not None and f_prev1 is not None and f_prev2.shape == f_prev1.shape == feature.shape:
        raw_features = [f_prev2, f_prev1, feature]
    elif f_prev1 is not None and f_prev1.shape == feature.shape:
        raw_features = [f_prev1, feature]
    else:
        # Not enough history with matching shapes — fall back to original
        derivative_approximation(cache_dic, current, feature)
        return

    # Smooth
    smooth_fn = exponential_smoothing if method == 'exponential' else moving_average_smoothing
    smoothed = smooth_fn(raw_features, alpha)

    # Compute derivatives from smoothed features
    updated_taylor_factors = {}
    updated_taylor_factors[0] = smoothed[-1]

    for i in range(cache_dic['max_order']):
        idx = len(smoothed) - 2 - i
        if idx < 0:
            break
        prev_val = smoothed[idx]
        if updated_taylor_factors[i].shape == prev_val.shape:
            updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - prev_val) / h
        else:
            break

    cache_dic["cache"][-1][s][l][m] = updated_taylor_factors


def derivative_approximation_hybrid_smoothing(cache_dic: Dict, current: Dict, feature: torch.Tensor,
                                              method: str = 'exponential', alpha: float = 0.8):
    """
    Hybrid smoothing: 1st order derivative uses raw features,
    2nd and higher order derivatives use smoothed features.
    """
    s, l, m = current["stream"], current["layer"], current["module"]
    h = current['activated_steps'][-1] - current['activated_steps'][-2]

    f_prev1 = cache_dic["cache"][-1][s][l][m].get(0, None)

    # 1st order: raw features
    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    if f_prev1 is not None and f_prev1.shape == feature.shape:
        updated_taylor_factors[1] = (feature - f_prev1) / h
    else:
        derivative_approximation(cache_dic, current, feature)
        return

    # 2nd+ order: smoothed features
    if cache_dic['max_order'] >= 2:
        f_prev2 = cache_dic["cache"][-2][s][l][m].get(0, None)
        if f_prev2 is not None and f_prev2.shape == feature.shape:
            raw_features = [f_prev2, f_prev1, feature]
            smooth_fn = exponential_smoothing if method == 'exponential' else moving_average_smoothing
            smoothed = smooth_fn(raw_features, alpha)

            d1_now = (smoothed[-1] - smoothed[-2]) / h
            d1_prev = (smoothed[-2] - smoothed[-3]) / h
            if d1_now.shape == d1_prev.shape:
                updated_taylor_factors[2] = (d1_now - d1_prev) / h

    cache_dic["cache"][-1][s][l][m] = updated_taylor_factors


def update_cache_or_approximate(cache_dic: Dict, current: Dict, feature=None):
    """
    Unified entry point for cache update (full mode) or Taylor approximation (taylor_cache mode).

    In full mode:
      1. If smoothing enabled, shift history first
      2. Init cache slot
      3. Compute derivatives (with optional smoothing)
      Returns the original feature.

    In taylor_cache mode:
      Returns taylor_formula result.
    """
    if current['type'] == 'full':
        use_smoothing = cache_dic.get('use_smoothing', False)
        use_hybrid = cache_dic.get('use_hybrid_smoothing', False)
        method = cache_dic.get('smoothing_method', 'exponential')
        alpha = cache_dic.get('smoothing_alpha', 0.8)

        # Order matters: shift history before init
        if use_smoothing:
            shift_cache_history(cache_dic, current)
        taylor_cache_init(cache_dic, current)

        if use_smoothing and use_hybrid:
            derivative_approximation_hybrid_smoothing(cache_dic, current, feature, method, alpha)
        elif use_smoothing:
            derivative_approximation_with_smoothing(cache_dic, current, feature, method, alpha)
        else:
            derivative_approximation(cache_dic, current, feature)

        return feature
    else:
        return taylor_formula(cache_dic, current)
