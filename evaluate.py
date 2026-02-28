import os
import torch
import cv2
import numpy as np
import re
import argparse
from PIL import Image
from tqdm import tqdm
import lpips
from skimage.metrics import structural_similarity as ssim
from transformers import CLIPProcessor, CLIPModel
import ImageReward as RM

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def load_prompts(prompt_file_path):
    """Load prompts from file"""
    with open(prompt_file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

def get_sorted_image_files(folder_path):
    """Get sorted image files from folder"""
    image_files = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
            image_files.append(filename)
    
    # Natural sort: extract all numbers from filename for robust ordering
    def natural_sort_key(filename):
        parts = re.split(r'(\d+)', filename)
        return [int(p) if p.isdigit() else p.lower() for p in parts]

    return sorted(image_files, key=natural_sort_key)

def calculate_psnr(img1, img2):
    """Calculate PSNR"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """Calculate SSIM"""
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        return ssim(gray1, gray2, data_range=255)
    return ssim(img1, img2, data_range=255)

def preprocess_for_lpips(img):
    """Preprocess image for LPIPS"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img * 2 - 1

def evaluate_all_metrics(test_folder, prompt_file_path=None, reference_folder=None, clip_model_path=None, imagereward_model_path=None):
    """Evaluate all metrics and return results"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert clip_model_path is not None
    assert imagereward_model_path is not None
    
    # Load models
    clip_model = CLIPModel.from_pretrained(clip_model_path)
    clip_model = clip_model.to(device)  # type: ignore
    clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
    
    # Load ImageReward model
    med_config = os.path.join(imagereward_model_path, "med_config.json")
    imagereward_path = os.path.join(imagereward_model_path, "ImageReward.pt")
    imagereward_model = RM.load(imagereward_path, download_root=imagereward_model_path, med_config=med_config).to(device)
    
    # LPIPS model
    lpips_model = lpips.LPIPS(net='alex', verbose=False).to(device)
    
    # Load data
    image_files = get_sorted_image_files(test_folder)
    prompts = load_prompts(prompt_file_path) if prompt_file_path else []
    ref_files = get_sorted_image_files(reference_folder) if reference_folder else []

    # Initialize score lists
    clip_scores = []
    imagereward_scores = []
    psnr_values = []
    ssim_values = []
    lpips_values = []

    # Process images
    for i, filename in enumerate(tqdm(image_files, desc="Evaluating")):
        try:
            img_path = os.path.join(test_folder, filename)
            img_pil = Image.open(img_path).convert("RGB")

            # CLIP Score and ImageReward (require prompts)
            if i < len(prompts):
                prompt = prompts[i]

                # CLIP Score
                with torch.no_grad():
                    inputs = clip_processor(text=prompt, images=img_pil, return_tensors="pt", padding=True, truncation=True).to(device)
                    outputs = clip_model(**inputs)
                    clip_scores.append(outputs.logits_per_image.item())

                # ImageReward (use built-in score() which handles preprocessing)
                if imagereward_model:
                    with torch.no_grad():
                        reward = imagereward_model.score(prompt, img_pil)
                        imagereward_scores.append(reward)

            # Quality metrics: match by sorted index, not filename
            if reference_folder and i < len(ref_files):
                ref_path = os.path.join(reference_folder, ref_files[i])
                img_cv = cv2.imread(img_path)
                ref_cv = cv2.imread(ref_path)

                if img_cv is not None and ref_cv is not None:
                    # Resize if needed
                    if img_cv.shape != ref_cv.shape:
                        ref_cv = cv2.resize(ref_cv, (img_cv.shape[1], img_cv.shape[0]))

                    # Calculate metrics
                    psnr_values.append(calculate_psnr(img_cv, ref_cv))
                    ssim_values.append(calculate_ssim(img_cv, ref_cv))

                    with torch.no_grad():
                        img_lpips = preprocess_for_lpips(img_cv).to(device)
                        ref_lpips = preprocess_for_lpips(ref_cv).to(device)
                        lpips_values.append(lpips_model(img_lpips, ref_lpips).item())

        except Exception as e:
            print(f"Warning: failed on {filename}: {e}")
            continue
    
    # Calculate averages
    results = {}
    if clip_scores:
        results['clip_score'] = np.mean(clip_scores)
    if imagereward_scores:
        results['imagereward'] = np.mean(imagereward_scores)
    if psnr_values:
        results['psnr'] = np.mean(psnr_values)
    if ssim_values:
        results['ssim'] = np.mean(ssim_values)
    if lpips_values:
        results['lpips'] = np.mean(lpips_values)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Unified metrics evaluation')
    parser.add_argument('--test_folder', type=str, required=True, help='Test images folder')
    parser.add_argument('--prompt_file', type=str, default="assets/prompts/DrawBench200.txt", help='Prompts file')
    parser.add_argument('--reference_folder', type=str, default=None, help='Reference images folder for quality metrics')
    # forexample, /data/public/models/laion/CLIP-ViT-g-14-laion2B-s12B-b42K
    parser.add_argument('--clip_model_path', type=str, default="/mnt/data0/pretrained_models/laion/CLIP-ViT-g-14-laion2B-s12B-b42K") 
    parser.add_argument('--imagereward_model_path', type=str, default="/mnt/data0/pretrained_models/zai-org/ImageReward")

    args = parser.parse_args()

    results = evaluate_all_metrics(
        test_folder=args.test_folder,
        prompt_file_path=args.prompt_file,
        reference_folder=args.reference_folder,
        clip_model_path=args.clip_model_path,
        imagereward_model_path=args.imagereward_model_path
    )
    
    # Output in requested format
    print("Result:(ClipScore, ImageReward, PSNR, SSIM, LPIPS)")
    print(f"{results.get('clip_score', 0):.4f}")
    print(f"{results.get('imagereward', 0):.4f}")
    print(f"{results.get('psnr', 0):.3f}")
    print(f"{results.get('ssim', 0):.4f}")
    print(f"{results.get('lpips', 0):.4f}")

if __name__ == "__main__":
    main()