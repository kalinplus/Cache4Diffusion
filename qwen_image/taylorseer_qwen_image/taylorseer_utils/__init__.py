import os
import json
from typing import Dict 
import torch
import math
import uuid
from collections import Counter

def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]
    #difference_distance = current['activated_times'][-1] - current['activated_times'][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for order in range(cache_dic['max_order']):
        # In cache[-1], select which stream, which layer and which module to update
        # 用有限差分公式算出下一阶的导数，updated_taylor_factors[order] 是当前步的导数， cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][order] 是上一个激活步的导数
        # if (cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].get(order, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
        cache_entry = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]
        if (cache_entry.get(order, None) is not None):
            updated_taylor_factors[order + 1] = (updated_taylor_factors[order] - cache_entry[order]) / difference_distance
        else:
            break
    
    cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = updated_taylor_factors

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
    cache_entry = cache[-1][current['stream']][current['layer']][current['module']]
    
    # 检查 cache[-2] 是否存在且包含所需数据
    has_cache_2 = (
        -2 in cache and
        current['stream'] in cache[-2] and
        current['layer'] in cache[-2][current['stream']] and
        current['module'] in cache[-2][current['stream']][current['layer']] and
        0 in cache[-2][current['stream']][current['layer']][current['module']]
    )
    
    if cache_entry.get(0, None) is not None and has_cache_2:
        F_0 = feature
        F_1 = cache_entry[0]
        F_2 = cache[-2][current['stream']][current['layer']][current['module']][0]
        
        # 三点向后差分：O(h²) 精度
        updated_taylor_factors[1] = (-3 * F_0 + 4 * F_1 - F_2) / (2 * h)
    elif cache_entry.get(0, None) is not None:
        # 回退到一阶差分
        updated_taylor_factors[1] = (feature - cache_entry[0]) / h
    
    for order in range(1, cache_dic['max_order']):
        if cache_entry.get(order, None) is not None:
            updated_taylor_factors[order + 1] = (updated_taylor_factors[order] - cache_entry[order]) / h
        else:
            break
    
    cache[-1][current['stream']][current['layer']][current['module']] = updated_taylor_factors

def shift_cache_history(cache_dic: Dict, current: Dict):
    """
    在每次调用 taylor_cache_init 之前，移动 current stream, layer, module 的缓存历史。
    """
    if (current['step'] == 0) and (cache_dic['taylor_cache']):
        cache_dic['cache'][-2][current['stream']][current['layer']][current['module']] = {}
    if (current['step'] > 0) and (cache_dic['taylor_cache']):
        cache_dic['cache'][-2][current['stream']][current['layer']][current['module']] = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]

def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor: 
    """
    Compute Taylor expansion error.
    
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
    Initialize Taylor cache and allocate storage for different-order derivatives in the Taylor cache.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if (current['step'] == 0) and (cache_dic['taylor_cache']):
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = {}

def taylor_formula_compare(cache_dic: Dict, current: Dict, ref_feature: torch.Tensor) -> torch.Tensor:
    """
    Compute Taylor expansion error from order 0 to cache_dic['max_order'] and record.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    :param ref_feature: Reference feature that forwards normally at the current step
    """
    x = current['step'] - current['activated_steps'][-1]
    # if current['layer'] == 0 and current['module'] == 'img_attn':
    #     print(f"[INFO] current activated_steps: {current['activated_steps']}")
    #     print(f"[INFO] Evaluating Taylor expansion at step {current['step']} (x={x}) for layer {current['layer']}, module {current['module']}")

    output = torch.zeros_like(ref_feature)
    errors = {}

    # 这里用的是长度，比如 max_order = 3，表示有 0,1,2 阶，共 3 阶
    max_order = len(cache_dic['cache'][-1][current['stream']][current['layer']][current['module']])
    for i in range(max_order):
        output += (1 / math.factorial(i)) * cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i] * (x ** i)
        error = torch.mean((output - ref_feature) ** 2).item()
        # print(f"Step {current['step']}, Layer {current['layer']}, Module {current['module']}, Max order {max_order}, MSE: {error}")
        errors[i] = error

    min_order = min(errors, key=errors.get)
    # print(f"Min MSE at order {min_order}, MSE: {errors[min_order]}\n")

    record = {
        "step": int(current['step']),
        "layer": int(current['layer']),
        "module": str(current['module']),
        "mse": {str(k): float(v) for k, v in errors.items()},  # 转为可 JSON 序列化的类型
        "best": int(min_order)
    }
    
    output_dir = "/data/huangkailin-20250908/Cache4Diffusion/qwen_image/visualize/test"
    output_json = os.path.join(output_dir, f"{cache_dic['prompt'][:30]}_order_eval.jsonl")
    with open(output_json, 'a') as f:
        f.write(json.dumps(record) + '\n')
    
    return output

def taylor_formula_layerwise(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    离线统计，在线使用的层级 Taylor 预测。
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    :return: Predicted feature with layerwise adaptive order
    """
    
    # 初始化存储结构
    if 'layerwise_best_orders' not in cache_dic:
        with open('qwen_image/visualize/aggregated_best_order.json', 'r') as f:
            cache_dic['layerwise_best_orders'] = json.load(f)
    
    step = current['step']
    layer = current['layer']
    module = current['module']
    
    if str(step) in cache_dic['layerwise_best_orders']["steps"]:
        # 使用离线的 best order 进行预测
        best_order = cache_dic['layerwise_best_orders']['steps'][str(step)]["modules"][module][layer]
        x = current['step'] - current['activated_steps'][-1]
        cache_entry = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]
        output = torch.zeros_like(cache_entry[0])
        if current['layer'] == 0 and current['module'] == 'txt_mlp':
            print(f"[Layerwise Adaptive] Using offline best order {best_order} at step {step}, layer {layer}, module {module} (x={x})")

        for i in range(best_order + 1):
            output += (1 / math.factorial(i)) * cache_entry[i] * (x ** i)
        return output    
    else:
        return taylor_formula(cache_dic, current)




def taylor_formula_layerwise_adaptive_online(cache_dic: Dict, current: Dict, 
                                            final_steps_high_order: int = 5,
                                            total_steps: int = 50) -> torch.Tensor:
    """
    在线自适应的层级 Taylor 预测。
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    :param final_steps_high_order: 最后多少步强制使用最高阶 (default: 5)
    :param total_steps: 总步数 (default: 50)
    :return: Predicted feature with layerwise adaptive order
    """
    
    # 初始化存储结构
    if 'layerwise_best_orders' not in cache_dic:
        cache_dic['layerwise_best_orders'] = {}
    
    layer = current['layer']
    module = current['module']
    layer_module_key = f"layer_{layer}_{module}"
    
    # 检查当前 layer+module 是否有已存储的 best order
    if layer_module_key not in cache_dic['layerwise_best_orders']:
        # 没有历史最佳 order，使用默认方法
        return taylor_formula(cache_dic, current)
    
    # 使用已有的 layerwise best order 进行预测
    return _predict_with_layerwise_order_online(cache_dic, current)

def update_layerwise_best_order_online(cache_dic: Dict, current: Dict, module: str, ref_feature: torch.Tensor):
    """
    在 first_enhance的最后一步/后续 activate steps，计算并更新每个 layer+module 的最佳 order。
    应该在真实前向传播完成后立即调用。
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    :param module: Module name ('img_attn', 'img_mlp', 'txt_attn', 'txt_mlp')
    :param ref_feature: Reference feature for this module (B, S, D)
    """
    # 判断是否是全量计算的最后一步
    if not is_last_activated_step(cache_dic, current):
        return
    
    # 初始化存储结构
    if 'layerwise_best_orders' not in cache_dic:
        cache_dic['layerwise_best_orders'] = {}
    
    # 检查是否有足够的历史信息
    if len(current['activated_steps']) < 2:
        print(f"[INFO] Not enough history at step {current['step']}, skipping layerwise update")
        return
    
    # 保存原始 module
    original_module = current.get('module')
    current['module'] = module
    
    # 使用倒数第二个 activated step 作为基准
    # 当前是倒数第一个 activated step
    x = current['step'] - current['activated_steps'][-2]
    
    cache_entry = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]
    
    max_available_order = len(cache_entry) - 1
    
    if max_available_order < 0:
        print(f"[WARNING] No cache available for layer {current['layer']}, module {module}")
        if original_module is not None:
            current['module'] = original_module
        return
    
    # 计算所有可用 order 的预测
    all_order_predictions = []
    cumulative_pred = torch.zeros_like(ref_feature)
    
    for order in range(max_available_order + 1):
        taylor_term = cache_entry[order] * (x ** order) / math.factorial(order)
        cumulative_pred = cumulative_pred + taylor_term
        all_order_predictions.append(cumulative_pred.clone())
    
    # 计算每个 order 的整体 MSE（对所有维度求平均）
    order_errors = []
    
    with torch.no_grad():
        for order_idx, pred in enumerate(all_order_predictions):
            # 计算整体 MSE
            mse = torch.mean((pred - ref_feature) ** 2).item()
            order_errors.append(mse)

    # 找出最佳 order
    best_order = int(torch.argmin(torch.tensor(order_errors)).item())
    
    # 存储到 cache_dic (按 layer+module 存储)
    layer = current['layer']
    layer_module_key = f"layer_{layer}_{module}"
    cache_dic['layerwise_best_orders'][layer_module_key] = best_order
    
    # 记录更新的 step
    cache_dic['layerwise_best_orders_step'] = current['step']
    
    # 恢复原始 module
    if original_module is not None:
        current['module'] = original_module
    
    # 打印统计信息（仅第一个 layer 打印，避免刷屏）
    if current.get('layer') == 0:
        update_count = cache_dic.get('layerwise_update_count', 0)
        # 前5次每次都打印，之后每5次打印一次
        if update_count < 5 or update_count % 5 == 0:
            print(f"\n[Layerwise Adaptive] Step {current['step']} (x={x}) - Best Order Distribution:")
            # 统计所有 layer 的 order 分布
            order_dist = Counter(cache_dic['layerwise_best_orders'].values())
            total = len(cache_dic['layerwise_best_orders'])
            dist_str = ", ".join([f"O={o}:{c}({c/total:.1%})" for o, c in sorted(order_dist.items())])
            print(f"  All layers: {dist_str}")
            
            # 打印当前更新的 layer+module 的详情
            print(f"  {layer_module_key}: O={best_order} (MSE={order_errors[best_order]:.6f})")
    
    cache_dic['layerwise_update_count'] = cache_dic.get('layerwise_update_count', 0) + 1
    
def _predict_with_layerwise_order_online(cache_dic: Dict, current: Dict) -> torch.Tensor:
    """
    使用已有的 layerwise best order 进行预测
    """
    x = current['step'] - current['activated_steps'][-1]
    cache_entry = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]
    
    base_feature = cache_entry[0]
    max_available_order = len(cache_entry) - 1
    
    # 获取当前 layer+module 的 best order
    layer = current['layer']
    module = current['module']
    layer_module_key = f"layer_{layer}_{module}"
    best_order = cache_dic['layerwise_best_orders'].get(layer_module_key, 0)
    
    # 限制 order 在可用范围内
    best_order = min(best_order, max_available_order)
    
    # 计算 Taylor 展开（只计算到 best_order）
    output = torch.zeros_like(base_feature)
    
    for order in range(best_order + 1):
        taylor_term = cache_entry[order] * (x ** order) / math.factorial(order)
        output = output + taylor_term
    
    return output

def is_last_activated_step(cache_dic: Dict, current: Dict) -> bool:
    """
    判断当前 step 是否是本轮全量计算的最后一步。
    
    逻辑：
    1. 当前 step 必须在 activated_steps 中（是全量计算步）
    2. 下一个 step 不在 activated_steps 中（即将开始预测阶段）
    """
    current_step = current['step']
    activated_steps = current['activated_steps']
    
    # 检查当前步是否是 activated step
    if current_step not in activated_steps:
        return False
    
    # 找到当前 step 在 activated_steps 中的位置
    try:
        current_idx = activated_steps.index(current_step)
    except ValueError:
        return False
    
    # 如果是 activated_steps 中的最后一个，认为是最后一步
    if current_idx == len(activated_steps) - 1:
        return True
    
    # 检查下一个 activated step 是否不是紧接着的
    next_activated = activated_steps[current_idx + 1]
    if next_activated - current_step > 1:
        # 说明中间有预测步，当前是这一轮的最后一个全量计算步
        return True
    
    return False
