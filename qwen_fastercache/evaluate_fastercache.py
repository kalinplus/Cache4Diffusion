"""
FasterCache 综合评测脚本
========================
输入:
  --test_folder   FasterCache 生成图像的文件夹（如 samples/fastercache/start1_interval8_alpha0.3）
  --log_file      run.log 或 flops_report.log 路径，用于提取 FLOPs / 加速比
  --prompt_file   prompts 文件，与生成时保持一致
  --imagereward_model_path  ImageReward 模型路径

输出（写入 eval_report.log，同时打印到终端）:
  Setting | ImageReward | FLOPs(T) | Speedup
"""

import os
import re
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import ImageReward as RM
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF


os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# ---------------------------------------------------------------------------
# 从 run.log 提取指定 setting 的 FLOPs / 加速比
# ---------------------------------------------------------------------------

def extract_flops_from_log(log_file: str, setting_tag: str):
    """
    在 log_file 中定位 'Setting: {setting_tag}' 段落，
    收集该段落内所有 [FasterCache FLOPs Report] 块，返回平均值。

    同一个 tag 可能在 log 中出现多次（生成阶段 + FLOPs 测量阶段），
    因此遍历所有匹配段，取第一个含有 FLOPs 数据的段落。

    返回 (avg_flops_T, avg_speedup) 或 (None, None) 若未找到。
    """
    if not os.path.exists(log_file):
        return None, None

    with open(log_file, 'r', errors='replace') as f:
        text = f.read()

    pattern_start = re.compile(
        r'Setting:\s*' + re.escape(setting_tag), re.IGNORECASE
    )
    next_setting  = re.compile(r'Setting:\s*\S', re.IGNORECASE)

    # 遍历该 tag 的所有出现位置，找到含 FLOPs 数据的那段
    search_pos = 0
    while True:
        m_start = pattern_start.search(text, search_pos)
        if m_start is None:
            break

        m_next  = next_setting.search(text, m_start.end())
        segment = text[m_start.start(): m_next.start()] if m_next else text[m_start.start():]

        flops_vals   = [float(x) for x in re.findall(r'Total FLOPs\s*:\s*([\d.]+)\s*T', segment)]
        speedup_vals = [float(x) for x in re.findall(r'Speedup\s*:\s*([\d.]+)x',         segment)]

        if flops_vals:
            return round(np.mean(flops_vals), 2), round(np.mean(speedup_vals), 2)

        search_pos = m_start.end()

    return None, None


# ---------------------------------------------------------------------------
# ImageReward 评测
# ---------------------------------------------------------------------------

def load_imagereward(model_path: str, device):
    med_config = os.path.join(model_path, "med_config.json")
    ckpt_path  = os.path.join(model_path, "ImageReward.pt")
    model = RM.load(ckpt_path, download_root=model_path, med_config=med_config)
    return model.to(device)


def get_sorted_images(folder: str):
    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def _key(f):
        m = re.search(r'(\d+)', f)
        return int(m.group(1)) if m else 0

    return sorted(files, key=_key)


def compute_imagereward(test_folder: str, prompts: list, model, device):
    reward_transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(
            (0.48145466, 0.4578275,  0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    image_files = get_sorted_images(test_folder)
    scores = []

    for i, fname in enumerate(tqdm(image_files, desc="ImageReward")):
        if i >= len(prompts):
            break
        try:
            img_pil    = Image.open(os.path.join(test_folder, fname)).convert("RGB")
            img_tensor = TF.pil_to_tensor(img_pil).unsqueeze(0).to(device)
            img_reward = reward_transform(img_tensor)

            inputs = model.blip.tokenizer(
                [prompts[i]], padding='max_length', truncation=True,
                max_length=512, return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                score = model.score_gard(inputs.input_ids, inputs.attention_mask, img_reward)
            scores.append(score.item())
        except Exception as e:
            print(f"  [skip] {fname}: {e}")

    return float(np.mean(scores)) if scores else float('nan'), len(scores)


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FasterCache 综合评测（ImageReward + FLOPs）")
    parser.add_argument("--test_folder",   type=str, required=True,
                        help="生成图像的文件夹，文件夹名即 setting tag")
    parser.add_argument("--log_file",      type=str, default="run.log",
                        help="run.log 路径，用于提取 FLOPs / 加速比")
    parser.add_argument("--prompt_file",   type=str,
                        default="/apdcephfs_cq11/share_300483685/jiachengliu/code/qwen_final/freqca/prompts/DrawBench200.txt",
                        help="与生成时相同的 prompt 文件")
    parser.add_argument("--imagereward_model_path", type=str,
                        default="/apdcephfs_zwfy8/share_304210317/jiachengliu/checkpoint/ImageReward",
                        help="ImageReward 模型目录")
    parser.add_argument("--output_log",    type=str, default="eval_report.log",
                        help="结果追加写入的 log 文件")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setting tag = 文件夹名
    setting_tag = os.path.basename(args.test_folder.rstrip("/"))
    print(f"\n{'='*60}")
    print(f"  Evaluating: {setting_tag}")
    print(f"{'='*60}")

    # 1. 从 log 提取 FLOPs 信息
    flops_T, speedup = extract_flops_from_log(args.log_file, setting_tag)
    if flops_T is not None:
        print(f"  [FLOPs]   {flops_T} T   Speedup {speedup}x")
    else:
        print(f"  [FLOPs]   not found in {args.log_file}")

    # 2. 加载 prompts
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        prompts = [l.strip() for l in f if l.strip()]

    # 3. 计算 ImageReward
    print("  Loading ImageReward model...")
    ir_model = load_imagereward(args.imagereward_model_path, device)

    ir_score, n_imgs = compute_imagereward(args.test_folder, prompts, ir_model, device)
    print(f"  [ImageReward]  {ir_score:.4f}  (n={n_imgs})")

    # 4. 输出汇总
    flops_str   = f"{flops_T}T"   if flops_T  is not None else "N/A"
    speedup_str = f"{speedup}x"  if speedup  is not None else "N/A"

    summary = (
        f"{setting_tag:<45} "
        f"IR={ir_score:>7.4f}  "
        f"FLOPs={flops_str:>10}  "
        f"Speedup={speedup_str:>6}  "
        f"n={n_imgs}"
    )
    print(f"\n  >> {summary}")

    # 5. 追加写入 eval_report.log
    header_needed = not os.path.exists(args.output_log)
    with open(args.output_log, 'a') as out:
        if header_needed:
            out.write(f"{'Setting':<45} {'ImageReward':>10}  {'FLOPs':>10}  {'Speedup':>8}  {'n':>5}\n")
            out.write("-" * 90 + "\n")
        out.write(
            f"{setting_tag:<45} "
            f"{ir_score:>10.4f}  "
            f"{flops_str:>10}  "
            f"{speedup_str:>8}  "
            f"{n_imgs:>5}\n"
        )
    print(f"  >> 结果已追加到 {args.output_log}")


if __name__ == "__main__":
    main()
