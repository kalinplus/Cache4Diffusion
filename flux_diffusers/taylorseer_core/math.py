"""
TaylorSeer core math functions.
Extracted from qwen_image version (most recent, includes TS_DEBUG_SHAPES support).
"""
import os
import math
import torch

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
        f"[branch={current.get('branch', '?')}]"
        f"[stream={current.get('stream', '?')}, layer={current.get('layer', '?')}, module={current.get('module', '?')}]"
    )


# ---------------------------------------------------------------------------
# Cache initialization
# ---------------------------------------------------------------------------

def module_cache_init(cache_dic, current):
    """Initialize Taylor cache slot for the current module (step 0 only)."""
    if (current['step'] == 0) and (cache_dic['taylor_cache']):
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = {}


# Alias used by Flux / HunyuanVideo
taylor_cache_init = module_cache_init


def cache_init(num_double_layers: int = 0, num_single_layers: int = 0,
               num_steps: int = 50, branches: dict = None):
    """
    Initialize cache dictionary for TaylorSeer caching.

    Args:
        num_double_layers: Number of double stream transformer layers
        num_single_layers: Number of single stream transformer layers
        num_steps: Total number of diffusion steps
        branches: Optional dict of {branch_name: num_layers}.
                  If provided, overrides num_double/single_layers.
                  E.g. {'cond': 60, 'uncond': 60} for CFG models,
                  or left None to use default double/single stream layout.

    Returns:
        cache_dic: Initialized cache dictionary
    """
    from .config import TaylorSeerConfig

    config = TaylorSeerConfig.from_env()

    # Build branch layout
    if branches is None:
        branches = {}
        if num_double_layers > 0:
            branches['double_stream'] = num_double_layers
        if num_single_layers > 0:
            branches['single_stream'] = num_single_layers

    cache = {}
    cache_index = {}
    cache[-1] = {}
    cache[-2] = {}  # For smoothing history
    cache_index[-1] = {}
    cache_index[-2] = {}
    cache_index['layer_index'] = {}

    # Attention map cache
    attn_map = {}
    attn_map[-1] = {}
    attn_map[-2] = {}

    # Initialize all branches
    for branch_name, num_layers in branches.items():
        cache[-1][branch_name] = {}
        cache[-2][branch_name] = {}
        attn_map[-1][branch_name] = {}
        attn_map[-2][branch_name] = {}

        # Determine module names based on branch type
        # single_stream uses 'total', everything else uses the 4-module layout
        if branch_name == 'single_stream':
            module_names = ['total']
        else:
            module_names = ['img_attn', 'img_mlp', 'txt_attn', 'txt_mlp']

        for j in range(num_layers):
            for i in [-1, -2]:
                cache[i][branch_name][j] = {}
                cache_index[i][j] = {}
                attn_map[i][branch_name][j] = {}
                for mod in module_names:
                    attn_map[i][branch_name][j][mod] = {}

    cache_dic = {
        'cache': cache,
        'cache_index': cache_index,
        'attn_map': attn_map,
        'cache_counter': 0,
        'cache_type': 'random',
        'fresh_ratio_schedule': 'ToCa',
        'fresh_ratio': 0.0,
        'fresh_threshold': config.cache_interval,
        'force_fresh': 'global',
        'soft_fresh_weight': 0.0,
        'taylor_cache': True,
        'max_order': config.max_order,
        'first_enhance': config.first_enhance,
        # Smoothing config
        'use_smoothing': config.use_smoothing,
        'use_hybrid_smoothing': config.use_hybrid_smoothing,
        'smoothing_method': config.smoothing_method,
        'smoothing_alpha': config.smoothing_alpha,
        'smoothed_derivatives': {},
    }

    return cache_dic


# ---------------------------------------------------------------------------
# Cache history shift (for smoothing)
# ---------------------------------------------------------------------------

def shift_cache_history(cache_dic, current):
    """Shift cache[-1] -> cache[-2] for smoothing."""
    if not cache_dic.get("taylor_cache", False):
        return

    cache = cache_dic["cache"]
    s, l, m = current["stream"], current["layer"], current["module"]

    if current["step"] == 0:
        cache[-2][s][l][m] = {}
        return

    cache[-2][s][l][m] = dict(cache[-1][s][l][m])


# ---------------------------------------------------------------------------
# Smoothing helpers
# ---------------------------------------------------------------------------

def exponential_smoothing(features: list, alpha: float) -> list:
    """Exponential smoothing over a list of tensors."""
    if len(features) <= 1:
        return features
    smoothed = [features[0]]
    for i in range(1, len(features)):
        smoothed.append(alpha * features[i] + (1 - alpha) * smoothed[i - 1])
    return smoothed


def moving_average_smoothing(features: list, window_size: int = 2) -> list:
    """Moving average smoothing over a list of tensors."""
    if len(features) < window_size:
        return features
    smoothed = []
    for i in range(len(features)):
        if i < window_size - 1:
            smoothed.append(sum(features[: i + 1]) / (i + 1))
        else:
            smoothed.append(sum(features[i - window_size + 1 : i + 1]) / window_size)
    return smoothed


# ---------------------------------------------------------------------------
# Derivative approximation
# ---------------------------------------------------------------------------

def derivative_approximation(cache_dic, current, feature):
    """Compute derivative approximation (no smoothing)."""
    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]

    updated = {0: feature}
    cache_entry = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]

    for i in range(cache_dic['max_order']):
        if cache_entry.get(i, None) is not None:
            updated[i + 1] = (updated[i] - cache_entry[i]) / difference_distance
        else:
            break

    s, l, m = current["stream"], current["layer"], current["module"]
    cache = cache_dic["cache"]
    cache[-2][s][l][m] = dict(cache[-1][s][l][m])
    cache[-1][s][l][m] = updated


def _collect_history_f0_fm1_fm2(cache_dic, current, feature):
    """Collect raw features: [F_{-2}, F_{-1}, F_0]."""
    cache = cache_dic["cache"]
    s, l, m = current["stream"], current["layer"], current["module"]
    entry_m1 = cache[-1][s][l][m]
    entry_m2 = cache[-2].get(s, {}).get(l, {}).get(m, {})
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
    alpha=0.8,
    window_size=2,
):
    h = current["activated_steps"][-1] - current["activated_steps"][-2]
    raw = _collect_history_f0_fm1_fm2(cache_dic, current, feature)

    if _TS_DEBUG_SHAPES:
        shapes = ", ".join(_shape_str(t) for t in raw)
        print(_debug_prefix(cache_dic, current), f"raw_len={len(raw)} raw_shapes=[{shapes}] alpha={alpha} method={smoothing_method}")

    if len(raw) >= 2:
        base = tuple(raw[-1].shape)
        mismatch = any(tuple(t.shape) != base for t in raw[:-1])
        if mismatch:
            msg = _debug_prefix(cache_dic, current) + " SHAPE_MISMATCH raw_shapes=[" + ", ".join(_shape_str(t) for t in raw) + "]"
            if _TS_STRICT_SHAPES:
                raise RuntimeError(msg)
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

    cache[-2][s][l][m] = dict(cache[-1][s][l][m])
    cache[-1][s][l][m] = updated


def derivative_approximation_hybrid_smoothing(
    cache_dic,
    current,
    feature,
    smoothing_method="exponential",
    alpha=0.8,
    window_size=2,
):
    """Hybrid smoothing: raw derivative for order-1, smoothed for order-2+."""
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

    cache[-2][s][l][m] = dict(cache[-1][s][l][m])
    cache[-1][s][l][m] = updated


# ---------------------------------------------------------------------------
# Taylor formula (inference / approximation)
# ---------------------------------------------------------------------------

def taylor_formula(cache_dic, current):
    """Compute Taylor expansion from cached derivatives."""
    x = current['step'] - current['activated_steps'][-1]
    cache_entry = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]
    output = 0
    for i in range(len(cache_entry)):
        output += (1 / math.factorial(i)) * cache_entry[i] * (x ** i)
    return output
