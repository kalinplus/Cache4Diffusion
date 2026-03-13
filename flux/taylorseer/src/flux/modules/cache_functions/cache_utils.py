import os
import torch
import math
from typing import Dict, List


def module_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache and allocate storage for different-order derivatives in the Taylor cache.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if current["step"] == 0:
        cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]] = {}


def shift_cache_history(cache_dic: Dict, current: Dict):
    """
    Shift cache history: move cache[-1] to cache[-2] for smoothing.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if not cache_dic.get("taylor_cache", False):
        return

    cache = cache_dic["cache"]
    s, l, m = current["stream"], current["layer"], current["module"]

    if current["step"] == 0:
        cache[-2][s][l][m] = {}  # Initialize as empty dict at first step
        return

    cache[-2][s][l][m] = cache[-1][s][l][m]  # Shift history


def exponential_smoothing(features: List[torch.Tensor], alpha: float) -> List[torch.Tensor]:
    """
    Apply exponential smoothing to a list of feature tensors.

    :param features: List of feature tensors [F_{-2}, F_{-1}, F_0]
    :param alpha: Smoothing coefficient (0-1), higher values keep more of the original
    :return: Smoothed feature tensors
    """
    if len(features) <= 1:
        return features

    smoothed = [features[0]]
    for i in range(1, len(features)):
        smoothed.append(alpha * features[i] + (1 - alpha) * smoothed[i - 1])
    return smoothed


def moving_average_smoothing(features: List[torch.Tensor], window_size: int = 2) -> List[torch.Tensor]:
    """
    Apply moving average smoothing to a list of feature tensors.

    :param features: List of feature tensors
    :param window_size: Size of the moving window (default: 2)
    :return: Smoothed feature tensors
    """
    if len(features) < window_size:
        return features

    smoothed = []
    for i in range(len(features)):
        if i < window_size - 1:
            smoothed.append(sum(features[: i + 1]) / (i + 1))
        else:
            smoothed.append(sum(features[i - window_size + 1 : i + 1]) / window_size)
    return smoothed


def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor | None):
    """
    Compute derivative approximation (original version without smoothing).

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    difference_distance = current["activated_steps"][-1] - current["activated_steps"][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic["max_order"]):
        if (
            cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]].get(i, None)
            is not None
        ) and (current["step"] > cache_dic["first_enhance"] - 2):
            updated_taylor_factors[i + 1] = (
                updated_taylor_factors[i]
                - cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]][i]
            ) / difference_distance
        else:
            break

    cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]] = updated_taylor_factors


def derivative_approximation_with_smoothing(
    cache_dic: Dict, current: Dict, feature: torch.Tensor | None, smoothing_method: str, alpha: float
):
    """
    Compute derivative approximation with smoothing (global smoothing mode).

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    :param feature: Current feature tensor
    :param smoothing_method: Smoothing method ('exponential' or 'moving_average')
    :param alpha: Smoothing coefficient
    """
    difference_distance = current["activated_steps"][-1] - current["activated_steps"][-2]
    cache = cache_dic["cache"]
    s, l, m = current["stream"], current["layer"], current["module"]

    # Collect history features: [F_{-2}, F_{-1}, F_0]
    raw_features = []
    if cache[-2][s][l][m].get(0, None) is not None:
        raw_features.append(cache[-2][s][l][m][0])  # F_{-2}
    if cache[-1][s][l][m].get(0, None) is not None:
        raw_features.append(cache[-1][s][l][m][0])  # F_{-1}
    raw_features.append(feature)  # F_0

    # Apply smoothing
    if len(raw_features) >= 2:
        if smoothing_method == "exponential":
            smoothed = exponential_smoothing(raw_features, alpha)
        else:  # moving_average
            smoothed = moving_average_smoothing(raw_features, window_size=2)
    else:
        # Not enough history, fall back to non-smoothing
        derivative_approximation(cache_dic, current, feature)
        return

    # Compute derivatives using smoothed features
    updated_taylor_factors = {}
    updated_taylor_factors[0] = smoothed[-1]  # Use smoothed current feature

    for i in range(cache_dic["max_order"]):
        if (cache[-1][s][l][m].get(i, None) is not None) and (
            current["step"] > cache_dic["first_enhance"] - 2
        ):
            if len(smoothed) >= 2:
                updated_taylor_factors[i + 1] = (smoothed[-1] - smoothed[-2]) / difference_distance
            else:
                updated_taylor_factors[i + 1] = (
                    updated_taylor_factors[i] - cache[-1][s][l][m][i]
                ) / difference_distance
        else:
            break

    cache[-1][s][l][m] = updated_taylor_factors


def derivative_approximation_hybrid_smoothing(
    cache_dic: Dict, current: Dict, feature: torch.Tensor | None, smoothing_method: str, alpha: float
):
    """
    Compute derivative approximation with hybrid smoothing:
    - First-order derivatives use original features
    - Second-order and higher derivatives use smoothed features

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    :param feature: Current feature tensor
    :param smoothing_method: Smoothing method ('exponential' or 'moving_average')
    :param alpha: Smoothing coefficient
    """
    difference_distance = current["activated_steps"][-1] - current["activated_steps"][-2]
    cache = cache_dic["cache"]
    s, l, m = current["stream"], current["layer"], current["module"]

    # Collect history features: [F_{-2}, F_{-1}, F_0]
    raw_features = []
    if cache[-2][s][l][m].get(0, None) is not None:
        raw_features.append(cache[-2][s][l][m][0])  # F_{-2}
    if cache[-1][s][l][m].get(0, None) is not None:
        raw_features.append(cache[-1][s][l][m][0])  # F_{-1}
    raw_features.append(feature)  # F_0

    # Initialize with zeroth-order
    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    # First-order derivative: use original features
    if len(raw_features) >= 2:
        updated_taylor_factors[1] = (raw_features[-1] - raw_features[-2]) / difference_distance
    elif cache[-1][s][l][m].get(0, None) is not None:
        updated_taylor_factors[1] = (feature - cache[-1][s][l][m][0]) / difference_distance
    else:
        # Cannot compute first-order derivative
        cache[-1][s][l][m] = updated_taylor_factors
        return

    # Second-order and higher derivatives: use smoothed features
    if cache_dic["max_order"] >= 2 and len(raw_features) >= 3:
        # Apply smoothing for higher-order computation
        if smoothing_method == "exponential":
            smoothed = exponential_smoothing(raw_features, alpha)
        else:  # moving_average
            smoothed = moving_average_smoothing(raw_features, window_size=2)

        # Compute first-order derivatives from smoothed features
        d1_now = None
        if len(smoothed) >= 2:
            d1_now = (smoothed[-1] - smoothed[-2]) / difference_distance
        if len(smoothed) >= 3:
            d1_prev = (smoothed[-2] - smoothed[-3]) / difference_distance
            if d1_now is not None:
                updated_taylor_factors[2] = (d1_now - d1_prev) / difference_distance
    elif cache[-1][s][l][m].get(1, None) is not None and cache_dic["max_order"] >= 2:
        # Fall back to original method for second-order
        updated_taylor_factors[2] = (updated_taylor_factors[1] - cache[-1][s][l][m][1]) / difference_distance

    cache[-1][s][l][m] = updated_taylor_factors


def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Compute Taylor expansion.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    x = current["step"] - current["activated_steps"][-1]

    output = cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]][0].clone() * 0

    for i in range(len(cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]])):
        output += (
            (1 / math.factorial(i))
            * cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]][i]
            * (x**i)
        )

    return output


def update_cache_or_approximate(cache_dic: Dict, current: Dict, feature: torch.Tensor | None = None):
    """
    Unified entry point for cache update (full compute) or Taylor approximation (cache mode).

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    :param feature: Feature tensor (required for full mode, None for cache mode)
    :return: feature for full mode, approximated output for cache mode
    """
    if current["type"] == "full":
        use_smoothing = cache_dic.get("use_smoothing", False)
        use_hybrid = cache_dic.get("use_hybrid_smoothing", False)
        smoothing_method = cache_dic.get("smoothing_method", "exponential")
        smoothing_alpha = cache_dic.get("smoothing_alpha", 0.8)

        # Order matters: shift history first, then init new slot
        if use_smoothing:
            shift_cache_history(cache_dic, current)
        module_cache_init(cache_dic, current)

        # Choose derivative computation method based on configuration
        if use_smoothing and use_hybrid:
            derivative_approximation_hybrid_smoothing(
                cache_dic, current, feature, smoothing_method, smoothing_alpha
            )
        elif use_smoothing:
            derivative_approximation_with_smoothing(
                cache_dic, current, feature, smoothing_method, smoothing_alpha
            )
        else:
            derivative_approximation(cache_dic, current, feature)

        return feature  # Return original feature in full mode
    else:
        return taylor_formula(cache_dic, current)  # Return Taylor approximation in cache mode
