import json
import os

def load_results(json_path):
    """加载 image_reward_results.json"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 转换为 {prompt_index: {'reward': reward, 'prompt': prompt}} 的字典
    return {r['prompt_index']: {'reward': r['reward'], 'prompt': r['prompt']} 
            for r in data['individual_results']}

def analyze_differences(exp_dir, naive_ts_dir, origin_dir=None, threshold=0.1):
    """
    分析 exp 相对于 naive_ts 的差异
    
    Args:
        exp_dir: exp/0.7 目录路径
        naive_ts_dir: naive_ts 目录路径
        origin_dir: origin 目录路径（可选）
        threshold: 差异阈值
    """
    exp_results = load_results(os.path.join(exp_dir, 'image_reward_results.json'))
    naive_results = load_results(os.path.join(naive_ts_dir, 'image_reward_results.json'))
    
    origin_results = None
    if origin_dir and os.path.exists(os.path.join(origin_dir, 'image_reward_results.json')):
        origin_results = load_results(os.path.join(origin_dir, 'image_reward_results.json'))
    
    improvements = []  # exp > naive_ts
    degradations = []  # exp < naive_ts
    
    for idx in sorted(exp_results.keys()):
        if idx not in naive_results:
            continue
            
        exp_reward = exp_results[idx]['reward']
        naive_reward = naive_results[idx]['reward']
        prompt = exp_results[idx]['prompt']
        
        diff = exp_reward - naive_reward
        
        record = {
            'index': idx,
            'prompt': prompt,
            'exp': exp_reward,
            'naive_ts': naive_reward,
            'diff': diff,
        }
        
        if origin_results and idx in origin_results:
            record['origin'] = origin_results[idx]['reward']
            record['exp_vs_origin'] = exp_reward - origin_results[idx]['reward']
            record['naive_vs_origin'] = naive_reward - origin_results[idx]['reward']
        
        if diff > threshold:
            improvements.append(record)
        elif diff < -threshold:
            degradations.append(record)
    
    # 按差异幅度排序
    improvements.sort(key=lambda x: x['diff'], reverse=True)
    degradations.sort(key=lambda x: x['diff'])
    
    # 输出结果
    print("=" * 100)
    print(f"EXP vs NAIVE_TS 分析 (阈值: {threshold})")
    print("=" * 100)
    
    print(f"\n【EXP 更好的 Prompts】(diff > {threshold}): {len(improvements)} 个\n")
    print(f"{'Idx':<5} {'Diff':>8} {'Exp':>8} {'Naive':>8} {'Origin':>8} Prompt")
    print("-" * 100)
    for r in improvements:
        prompt_short = r['prompt'][:200] + '...' if len(r['prompt']) > 200 else r['prompt']
        origin_str = f"{r.get('origin', 0):>8.4f}" if 'origin' in r else "    N/A "
        print(f"{r['index']:<5} {r['diff']:>+8.4f} {r['exp']:>8.4f} {r['naive_ts']:>8.4f} {origin_str} {prompt_short}")
    
    print(f"\n\n【NAIVE_TS 更好的 Prompts】(diff < -{threshold}): {len(degradations)} 个\n")
    print(f"{'Idx':<5} {'Diff':>8} {'Exp':>8} {'Naive':>8} {'Origin':>8} Prompt")
    print("-" * 100)
    for r in degradations:
        prompt_short = r['prompt'][:200] + '...' if len(r['prompt']) > 200 else r['prompt']
        origin_str = f"{r.get('origin', 0):>8.4f}" if 'origin' in r else "    N/A "
        print(f"{r['index']:<5} {r['diff']:>+8.4f} {r['exp']:>8.4f} {r['naive_ts']:>8.4f} {origin_str} {prompt_short}")
    
    # 统计摘要
    print(f"\n\n{'=' * 100}")
    print("统计摘要")
    print("=" * 100)
    
    all_diffs = [exp_results[i]['reward'] - naive_results[i]['reward'] 
                 for i in exp_results.keys() if i in naive_results]
    
    print(f"总共比较: {len(all_diffs)} 个 prompts")
    print(f"EXP 更好 (diff > {threshold}): {len(improvements)} 个")
    print(f"NAIVE_TS 更好 (diff < -{threshold}): {len(degradations)} 个")
    print(f"差异不大 (|diff| <= {threshold}): {len(all_diffs) - len(improvements) - len(degradations)} 个")
    print(f"\n平均差异: {sum(all_diffs) / len(all_diffs):+.4f}")
    print(f"最大提升: {max(all_diffs):+.4f}")
    print(f"最大下降: {min(all_diffs):+.4f}")
    
    # 按 prompt 类型分析
    print(f"\n\n{'=' * 100}")
    print("按 Prompt 类型分析")
    print("=" * 100)
    
    categories = {
        'color': (0, 15),      # A red/black/pink colored ...
        'composition': (15, 35),  # A X and a Y ...
        'counting': (35, 55),   # One/Two/Three ...
        'spatial': (114, 134),  # on top of / underneath / left / right
        'text': (179, 200),     # storefront / sign / fireworks with text
    }
    
    for cat_name, (start, end) in categories.items():
        cat_diffs = [exp_results[i]['reward'] - naive_results[i]['reward'] 
                     for i in range(start, end) if i in exp_results and i in naive_results]
        if cat_diffs:
            avg_diff = sum(cat_diffs) / len(cat_diffs)
            print(f"{cat_name:15s} (idx {start:3d}-{end-1:3d}): 平均差异 {avg_diff:+.4f}")
    
    return improvements, degradations


if __name__ == "__main__":
    base_dir = "/data/huangkailin-20250908/Cache4Diffusion/flux/outputs/smooth/db200"
    
    exp_dir = os.path.join(base_dir, "exp/0.7")
    naive_ts_dir = os.path.join(base_dir, "naive_ts")
    origin_dir = os.path.join(base_dir, "origin")
    
    # 分析差异，阈值设为 0.1
    improvements, degradations = analyze_differences(
        exp_dir=exp_dir,
        naive_ts_dir=naive_ts_dir,
        origin_dir=origin_dir,
        threshold=0.1
    )