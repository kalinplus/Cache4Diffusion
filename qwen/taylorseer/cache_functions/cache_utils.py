import os
import torch
import math
from typing import Dict, Tuple

_TS_DEBUG_SMOOTH = os.environ.get("TS_DEBUG_SMOOTH", "0").lower() in ("1", "true", "yes")

# Filter: e.g. "0,img_attn" to only print layer 0 img_attn
_TS_DEBUG_FILTER = os.environ.get("TS_DEBUG_FILTER", "").strip()


def _should_debug(current):
    if not _TS_DEBUG_FILTER:
        return True
    parts = [p.strip() for p in _TS_DEBUG_FILTER.split(",")]
    layer = str(current.get('layer', ''))
    module = str(current.get('module', ''))
    # Accept "layer,module" or just "layer"
    if len(parts) == 2:
        return layer == parts[0] and module == parts[1]
    return layer == parts[0]


def _debug_prefix(current):
    return f"[TS][step={current.get('step')}][{current.get('stream')}/{current.get('layer')}/{current.get('module')}]"


# ─── Smoothing utilities ────────────────────────────────────────────────────────

def exponential_smoothing(features, alpha):
    """Apply exponential smoothing to a list of tensors."""
    if len(features) <= 1:
        return features
    smoothed = [features[0]]
    for i in range(1, len(features)):
        smoothed.append(alpha * features[i] + (1 - alpha) * smoothed[i - 1])
    return smoothed


def moving_average_smoothing(features, window_size=2):
    """Apply moving average smoothing to a list of tensors."""
    if len(features) < window_size:
        return features
    smoothed = []
    for i in range(len(features)):
        if i < window_size - 1:
            smoothed.append(sum(features[: i + 1]) / (i + 1))
        else:
            smoothed.append(sum(features[i - window_size + 1 : i + 1]) / window_size)
    return smoothed


def get_smoothed_features(cache_dic, current):
    """
    Collect raw features from cache[-2] and cache[-1] for the current stream/layer/module,
    then apply smoothing.
    Returns list: [F_{-2}, F_{-1}, F_0(current)] or None if history is incomplete.
    """
    s, l, m = current['stream'], current['layer'], current['module']
    max_order = cache_dic.get('max_order', 1)
    method = cache_dic.get('smoothing_method', 'exponential')
    alpha = cache_dic.get('smoothing_alpha', 0.8)

    # Need at least order 0 (feature) in both cache[-2] and cache[-1]
    if s not in cache_dic['cache'][-1] or l not in cache_dic['cache'][-1][s]:
        return None

    cache_prev = cache_dic['cache'][-1][s][l][m]
    cache_prev2 = cache_dic['cache'][-2].get(s, {}).get(l, {}).get(m, {})

    if cache_prev.get(0, None) is None or cache_prev2.get(0, None) is None:
        return None

    # Collect F_{-2} and F_{-1} (0th-order features)
    f_prev2 = cache_prev2[0]
    f_prev1 = cache_prev[0]

    # Build raw feature list — shapes must match
    if f_prev2.shape != f_prev1.shape:
        return None

    raw = [f_prev2, f_prev1]
    return raw


def shift_cache_history(cache_dic, current):
    """
    Shift cache[-1] → cache[-2] before a new full compute step.
    This preserves the previous step's cache for smoothing derivative computation.
    """
    s, l, m = current['stream'], current['layer'], current['module']

    if current['step'] == 0:
        # First step: initialize empty history
        if s not in cache_dic['cache'][-2]:
            cache_dic['cache'][-2][s] = {}
        if l not in cache_dic['cache'][-2][s]:
            cache_dic['cache'][-2][s][l] = {}
        cache_dic['cache'][-2][s][l][m] = {}
        return

    # Move current cache to history (shallow copy of dict, tensors are read-only here)
    if s not in cache_dic['cache'][-2]:
        cache_dic['cache'][-2][s] = {}
    if l not in cache_dic['cache'][-2][s]:
        cache_dic['cache'][-2][s][l] = {}
    cache_dic['cache'][-2][s][l][m] = dict(cache_dic['cache'][-1][s][l][m])


# ─── Derivative approximation ────────────────────────────────────────────────

def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation.
    """
    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic['max_order']):
        if (cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].get(i, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
            updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i]) / difference_distance
        else:
            break

    s, l, m = current['stream'], current['layer'], current['module']
    cache_dic['cache'][-2][s][l][m] = dict(cache_dic['cache'][-1][s][l][m])
    cache_dic['cache'][-1][s][l][m] = updated_taylor_factors
    if _TS_DEBUG_SMOOTH and _should_debug(current):
        print(_debug_prefix(current), f"deriv_approx: saved to cache[-2] keys={list(cache_dic['cache'][-2][s][l][m].keys())}, cache[-1] keys={list(updated_taylor_factors.keys())}")


def derivative_approximation_with_smoothing(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation using smoothed features.
    Uses cache[-2] and cache[-1] to collect a short history, applies smoothing,
    then computes derivatives from the smoothed trajectory.
    """
    s, l, m = current['stream'], current['layer'], current['module']
    max_order = cache_dic.get('max_order', 1)
    method = cache_dic.get('smoothing_method', 'exponential')
    alpha = cache_dic.get('smoothing_alpha', 0.8)

    # Collect raw features [F_{-2}, F_{-1}, F_0]
    raw = get_smoothed_features(cache_dic, current)
    if raw is None:
        # Fall back to non-smoothed
        derivative_approximation(cache_dic, current, feature)
        return

    raw.append(feature)
    if _TS_DEBUG_SMOOTH and _should_debug(current):
        shapes = [tuple(t.shape) for t in raw]
        ids = [id(t) for t in raw]
        print(_debug_prefix(current), f"smooth: raw_len={len(raw)} shapes={shapes} ids={ids} id_dup={len(set(ids)) < len(ids)}")

    # Check shape consistency across all three
    if raw[0].shape != raw[1].shape or raw[1].shape != raw[2].shape:
        derivative_approximation(cache_dic, current, feature)
        return

    # Apply smoothing
    if method == 'moving_average':
        smoothed = moving_average_smoothing(raw, window_size=2)
    else:
        smoothed = exponential_smoothing(raw, alpha)

    # Compute derivatives from smoothed trajectory
    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]
    h = difference_distance

    updated_taylor_factors = {}
    updated_taylor_factors[0] = smoothed[-1]  # F'_0

    # First-order derivative from smoothed features
    if len(smoothed) >= 2 and current['step'] > cache_dic['first_enhance'] - 2:
        updated_taylor_factors[1] = (smoothed[-1] - smoothed[-2]) / h
    else:
        cache_dic['cache'][-1][s][l][m] = updated_taylor_factors
        return

    # Higher-order derivatives recursively, same structure as non-smoothed version
    for i in range(1, max_order):
        prev_cache = cache_dic['cache'][-2].get(s, {}).get(l, {}).get(m, {})
        prev_factor = prev_cache.get(i, None)
        if prev_factor is not None and current['step'] > cache_dic['first_enhance'] - 2:
            updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - prev_factor) / h
        else:
            break

    cache_dic['cache'][-2][s][l][m] = dict(cache_dic['cache'][-1][s][l][m])
    cache_dic['cache'][-1][s][l][m] = updated_taylor_factors
    if _TS_DEBUG_SMOOTH and _should_debug(current):
        print(_debug_prefix(current), f"smooth: saved to cache[-2] keys={list(cache_dic['cache'][-2][s][l][m].keys())}, cache[-1] keys={list(updated_taylor_factors.keys())}")


# ─── Taylor formula ───────────────────────────────────────────────────────────

def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    Compute Taylor expansion approximation.
    """
    x = current['step'] - current['activated_steps'][-1]

    factors = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]

    output = factors[0].clone() * 0

    for i in range(len(factors)):
        output += (1 / math.factorial(i)) * factors[i] * (x ** i)

    return output


# ─── Module cache init ────────────────────────────────────────────────────────

def module_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache for a new module.
    """
    s, l, m = current['stream'], current['layer'], current['module']
    if current['step'] == 0:
        cache_dic['cache'][-1][s][l][m] = {}


# ─── Unified entry point ─────────────────────────────────────────────────────

def update_cache_or_approximate(cache_dic: Dict, current: Dict, feature: torch.Tensor = None):
    """
    Unified entry for both full and cache modes.
    - full mode: shift history, init module, compute derivatives (smoothed or not), return original feature
    - cache mode: return Taylor-approximated feature
    """
    if current['type'] == 'full':
        use_smoothing = cache_dic.get('use_smoothing', False)
        if _TS_DEBUG_SMOOTH and _should_debug(current):
            print(_debug_prefix(current), f"update_cache: type=full use_smoothing={use_smoothing}")

        module_cache_init(cache_dic, current)

        if use_smoothing:
            derivative_approximation_with_smoothing(cache_dic, current, feature)
        else:
            derivative_approximation(cache_dic, current, feature)

        return feature  # full mode returns original feature
    else:
        if _TS_DEBUG_SMOOTH and _should_debug(current):
            print(_debug_prefix(current), "update_cache: type=Taylor (approximate)")
        return taylor_formula(cache_dic, current)  # cache mode returns approximate


def pipeline_with_cache(pipe):

    import types
    from pipeline.transformer_qwenimage import QwenImageTransformer2DModel as LocalQwenImageTransformer2DModel
    from pipeline.transformer_qwenimage import QwenImageTransformerBlock as LocalQwenImageTransformerBlock

    pipe.transformer.forward = types.MethodType(LocalQwenImageTransformer2DModel.forward, pipe.transformer)

    for _, block in enumerate(pipe.transformer.transformer_blocks):
        block.forward = types.MethodType(LocalQwenImageTransformerBlock.forward, block)

    return pipe