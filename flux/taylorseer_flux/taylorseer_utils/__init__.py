from typing import Dict 
import torch
import math

def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for order in range(cache_dic['max_order']):
        cache_entry = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]
        if (cache_entry.get(order, None) is not None):
            updated_taylor_factors[order + 1] = (updated_taylor_factors[order] - cache_entry[order]) / difference_distance
        else:
            break
    
    cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = updated_taylor_factors


def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor: 
    """
    Compute Taylor expansion.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    x = current['step'] - current['activated_steps'][-1]
    output = 0

    for i in range(len(cache_dic['cache'][-1][current['stream']][current['layer']][current['module']])):
        output += (1 / math.factorial(i)) * cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i] * (x ** i)
    
    return output


def taylor_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache and allocate storage for different-order derivatives.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if (current['step'] == 0) and (cache_dic['taylor_cache']):
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = {}


# ==================== 平滑滤波函数 ====================

def exponential_smoothing(features: list, alpha: float) -> list:
    """
    指数平滑滤波：对历史特征序列进行平滑处理。
    :param features: 特征列表 [x1, x2, x3, ...]，每个是 (N, D) 的 tensor，历史特征在前
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


# ==================== 缓存历史移动 ====================

def shift_cache_history(cache_dic: Dict, current: Dict):
    """
    在每次调用 taylor_cache_init 之前，移动 current stream, layer, module 的缓存历史。
    cache[-1] -> cache[-2]
    """
    if (current['step'] == 0) and (cache_dic['taylor_cache']):
        cache_dic['cache'][-2][current['stream']][current['layer']][current['module']] = {}
    if (current['step'] > 0) and (cache_dic['taylor_cache']):
        cache_dic['cache'][-2][current['stream']][current['layer']][current['module']] = \
            cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]


# ==================== 带平滑的导数近似 ====================

def derivative_approximation_with_smoothing(cache_dic: Dict, current: Dict, feature: torch.Tensor, 
                                            smoothing_method: str = 'moving_average', alpha: float = 0.7):
    """
    带平滑预处理的导数近似计算。
    需要配合 shift_cache_history 使用以维护 cache[-2]。
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    :param feature: 当前特征
    :param smoothing_method: 'exponential' 或 'moving_average'
    :param alpha: 指数平滑系数
    """
    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]
    
    cache = cache_dic['cache']
    stream, layer, module = current['stream'], current['layer'], current['module']
    cache_entry = cache[-1][stream][layer][module]
    
    # ========== 收集历史特征 ==========
    # 顺序: [F_{-2}, F_{-1}, F_0] (从旧到新)
    history_features = []
    
    # F_{-2}: 从 cache[-2] 获取
    has_cache_2 = (
        -2 in cache and
        stream in cache[-2] and
        layer in cache[-2][stream] and
        module in cache[-2][stream][layer] and
        0 in cache[-2][stream][layer][module]
    )
    if has_cache_2:
        history_features.append(cache[-2][stream][layer][module][0])
    
    # F_{-1}: 从 cache[-1] 获取
    if cache_entry.get(0, None) is not None:
        history_features.append(cache_entry[0])
    
    # F_0: 当前特征
    history_features.append(feature)
    
    # ========== 根据历史长度选择策略 ==========
    updated_taylor_factors = {}
    
    if len(history_features) >= 3:
        # 有3个点，可以做平滑
        if smoothing_method == 'exponential':
            smoothed = exponential_smoothing(history_features, alpha)
        elif smoothing_method == 'moving_average':
            smoothed = moving_average_smoothing(history_features, window_size=2)
        else:
            smoothed = history_features
        
        # 使用平滑后的值
        updated_taylor_factors[0] = smoothed[-1]  # 平滑后的当前
        updated_taylor_factors[1] = (smoothed[-1] - smoothed[-2]) / difference_distance
        
    elif len(history_features) == 2:
        # 只有2个点，无法平滑，使用普通差分
        updated_taylor_factors[0] = feature
        updated_taylor_factors[1] = (feature - history_features[0]) / difference_distance
        
    else:
        # 只有1个点（第一步），只存0阶
        updated_taylor_factors[0] = feature
    
    # ========== 高阶导数递推 ==========
    for order in range(1, cache_dic['max_order']):
        if cache_entry.get(order, None) is not None and (order) in updated_taylor_factors:
            updated_taylor_factors[order + 1] = (updated_taylor_factors[order] - cache_entry[order]) / difference_distance
        else:
            break
    
    cache[-1][stream][layer][module] = updated_taylor_factors


# ==================== 混合平滑差分 ====================

def derivative_approximation_hybrid_smoothing(cache_dic: Dict, current: Dict, feature: torch.Tensor,
                                               smoothing_method: str = 'moving_average',
                                               alpha: float = 0.5,
                                               window_size: int = 2):
    """
    移动平均混合差分：
    - 0阶项：使用原始值（保证起点精确）
    - 1阶导数：使用原始序列计算（保留主要趋势）
    - 高阶导数：使用平滑后的序列计算（抑制噪声放大）
    
    需要配合 shift_cache_history 使用以维护 cache[-2]。
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    :param feature: 当前特征
    :param smoothing_method: 'exponential' 或 'moving_average'
    :param alpha: 指数平滑系数
    :param window_size: 移动平均窗口大小
    """
    h = current['activated_steps'][-1] - current['activated_steps'][-2]
    
    cache = cache_dic['cache']
    stream, layer, module = current['stream'], current['layer'], current['module']
    cache_entry = cache[-1][stream][layer][module]
    
    # ========== 收集原始历史特征 ==========
    raw_features = []
    
    # F_{-2}
    has_cache_2 = (
        -2 in cache and
        stream in cache[-2] and
        layer in cache[-2][stream] and
        module in cache[-2][stream][layer] and
        0 in cache[-2][stream][layer][module]
    )
    if has_cache_2:
        raw_features.append(cache[-2][stream][layer][module][0])
    
    # F_{-1}
    if cache_entry.get(0, None) is not None:
        raw_features.append(cache_entry[0])
    
    # F_0
    raw_features.append(feature)
    
    # ========== 根据历史长度计算 ==========
    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature  # 0阶：原始值
    
    if len(raw_features) == 1:
        # 只有当前点，无法计算导数
        cache[-1][stream][layer][module] = updated_taylor_factors
        return
    
    # 1阶导数：使用原始值（两点差分）
    updated_taylor_factors[1] = (raw_features[-1] - raw_features[-2]) / h
    
    if len(raw_features) >= 3 and cache_dic['max_order'] >= 2:
        # ========== 计算平滑序列 ==========
        if smoothing_method == 'exponential':
            smoothed = exponential_smoothing(raw_features, alpha)
        else:
            smoothed = moving_average_smoothing(raw_features, window_size)
        
        # 2阶导数：使用平滑序列的1阶导数差分
        smoothed_d1_current = (smoothed[-1] - smoothed[-2]) / h
        smoothed_d1_prev = (smoothed[-2] - smoothed[-3]) / h
        
        # 2阶 = (1阶_current - 1阶_prev) / h
        updated_taylor_factors[2] = (smoothed_d1_current - smoothed_d1_prev) / h
        
        # 3阶及更高：基于平滑导数递推
        for order in range(2, cache_dic['max_order']):
            if cache_entry.get(order, None) is not None and order in updated_taylor_factors:
                updated_taylor_factors[order + 1] = (updated_taylor_factors[order] - cache_entry[order]) / h
            else:
                break
    else:
        # 只有2个点，使用普通递推
        for order in range(1, cache_dic['max_order']):
            if cache_entry.get(order, None) is not None and order in updated_taylor_factors:
                updated_taylor_factors[order + 1] = (updated_taylor_factors[order] - cache_entry[order]) / h
            else:
                break
    
    cache[-1][stream][layer][module] = updated_taylor_factors


# ==================== 三点混合差分（高精度，但是实践证明效果很差） ====================

def derivative_approximation_hybrid(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    混合策略：
    - 1阶导数用三点差分（从特征计算，精度 O(h²)）
    - 2阶及以上用递推差分（从导数计算，精度 O(h)）
    """
    activated_steps = current['activated_steps']
    
    if len(activated_steps) < 3:
        derivative_approximation(cache_dic, current, feature)
        return
    
    h = activated_steps[-1] - activated_steps[-2]
    
    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature  # 0阶
    
    cache = cache_dic['cache']
    stream, layer, module = current['stream'], current['layer'], current['module']
    cache_entry = cache[-1][stream][layer][module]
    
    # 检查 cache[-2] 是否存在且包含所需数据
    has_cache_2 = (
        -2 in cache and
        stream in cache[-2] and
        layer in cache[-2][stream] and
        module in cache[-2][stream][layer] and
        0 in cache[-2][stream][layer][module]
    )
    
    if cache_entry.get(0, None) is not None and has_cache_2:
        F_0 = feature
        F_1 = cache_entry[0]
        F_2 = cache[-2][stream][layer][module][0]
        
        # 三点向后差分：O(h²) 精度
        updated_taylor_factors[1] = (3 * F_0 - 4 * F_1 + F_2) / (2 * h)
    elif cache_entry.get(0, None) is not None:
        # 回退到一阶差分
        updated_taylor_factors[1] = (feature - cache_entry[0]) / h
    
    for order in range(1, cache_dic['max_order']):
        if cache_entry.get(order, None) is not None and order in updated_taylor_factors:
            updated_taylor_factors[order + 1] = (updated_taylor_factors[order] - cache_entry[order]) / h
        else:
            break
    
    cache[-1][stream][layer][module] = updated_taylor_factors