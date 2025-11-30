import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import json
import torch
from PIL import Image
import ImageReward as RM
import re

def load_prompts_from_txt(prompts_file):
    """
    从 txt 文件加载 prompts，每行一个
    
    Args:
        prompts_file: txt 文件路径
    
    Returns:
        list: prompts 列表，索引对应行号
    """
    prompts = []
    with open(prompts_file, 'r', encoding='utf-8') as f:
        for line in f:
            prompts.append(line.strip())
    return prompts

def extract_prompt_index(filename):
    """
    从文件名中提取 prompt 编号
    例如: TaylorSeer_0000_xxx.png -> 0
         TaylorSeer_0123_xxx.png -> 123
    
    Args:
        filename: 图像文件名
    
    Returns:
        int: prompt 编号，如果提取失败返回 None
    """
    # 匹配类似 TaylorSeer_0000_ 的模式
    match = re.search(r'_(\d{4})_', filename)
    if match:
        return int(match.group(1))
    return None

def evaluate_image_reward(image_dir, prompts_file=None, recent_n=10):
    """
    评估指定目录下图像的 Image Reward 分数
    
    Args:
        image_dir: 图像目录路径
        prompts_file: prompts txt 文件路径
    """
    # 初始化 ImageReward 模型
    print("Loading ImageReward model...")
    model = RM.load("ImageReward-v1.0")
    
    # 获取所有图像文件
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png'))])
    # print(image_files)
    
    # 加载 prompts
    prompts = []
    if prompts_file and os.path.exists(prompts_file):
        prompts = load_prompts_from_txt(prompts_file)
        print(f"Loaded {len(prompts)} prompts from {prompts_file}")
    else:
        print("Warning: No prompts file provided or file not found!")
    # print(prompts)
    
    results = []
    total_reward = 0
    matched_count = 0
    unmatched_files = []
    
    print(f"\nEvaluating {len(image_files)} images...")
    
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)
        
        # 从文件名提取 prompt 编号
        prompt_idx = extract_prompt_index(img_file)
        
        if prompt_idx is None:
            print(f"Warning: Cannot extract prompt index from {img_file}")
            unmatched_files.append(img_file)
            continue
        
        # 获取对应的 prompt
        if prompt_idx >= len(prompts):
            print(f"Warning: Prompt index {prompt_idx} out of range for {img_file}")
            unmatched_files.append(img_file)
            continue
        
        prompt = prompts[prompt_idx]
        # print(f"Processing {img_file}: extracted prompt index {prompt_idx}, prompt: {prompt}")
        
        try:
            # 加载图像
            image = Image.open(img_path).convert('RGB')
            
            # 计算 reward
            with torch.no_grad():
                reward = model.score(prompt, image)
            
            results.append({
                'image': img_file,
                'prompt_index': prompt_idx,
                'prompt': prompt,
                'reward': float(reward)
            })
            
            total_reward += reward
            matched_count += 1
            
            if (matched_count) % recent_n == 0:
                recent_n_rewards = [res['reward'] for res in results[-recent_n:]]
                avg_recent_n = sum(recent_n_rewards) / recent_n
                # print(f"Processed {matched_count}/{len(image_files)}, Last {recent_n} Avg Reward: {avg_recent_n:.4f}")
                print(f"{avg_recent_n:.4f}")
                
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
    
    # 计算平均值
    avg_reward = total_reward / matched_count if matched_count > 0 else 0
    
    # 保存结果
    output_file = os.path.join(image_dir, 'image_reward_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'average_reward': float(avg_reward),
            'total_images': len(image_files),
            'matched_images': matched_count,
            'unmatched_files': unmatched_files,
            'individual_results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"Total images: {len(image_files)}")
    print(f"Successfully matched: {matched_count}")
    print(f"Unmatched: {len(unmatched_files)}")
    if unmatched_files:
        print(f"Unmatched files: {unmatched_files[:5]}{'...' if len(unmatched_files) > 5 else ''}")
    print(f"\nAverage Image Reward: {avg_reward:.4f}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")
    
    return avg_reward, results

if __name__ == "__main__":
    # 设置图像目录
    # image_dir = "/data/huangkailin-20250908/Cache4Diffusion/flux/outputs/smooth/db200/naive_ts"
    image_dir = "/data/huangkailin-20250908/Cache4Diffusion/flux/outputs/smooth/db200/exp/0.7"
    # image_dir = "/data/huangkailin-20250908/Cache4Diffusion/flux/outputs/smooth/db200/origin"
    
    # 指定 prompts txt 文件路径
    prompts_file = "/data/huangkailin-20250908/Cache4Diffusion/assets/prompts/DrawBench200.txt"
    
    # 运行评估
    avg_reward, results = evaluate_image_reward(image_dir, prompts_file)