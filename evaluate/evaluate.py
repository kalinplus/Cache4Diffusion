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

# Default eval-model locations. The legacy defaults under /mnt/data0/... do not
# exist on this host, so point at the locally-downloaded copies and allow an
# override through env vars / CLI. CLIP defaults to the OpenAI ViT-L/14 that is
# present locally (laion/CLIP-ViT-g-14 is not cached here); switch back with
# --clip_model_path or EVAL_CLIP_MODEL_PATH.
DEFAULT_CLIP_MODEL_PATH = os.environ.get(
    "EVAL_CLIP_MODEL_PATH",
    "/mnt/workspace/hkl/models/openai/clip-vit-large-patch14",
)
DEFAULT_IMAGEREWARD_MODEL_PATH = os.environ.get(
    "EVAL_IMAGEREWARD_MODEL_PATH",
    "/mnt/workspace/hkl/models/zai-org/ImageReward",
)


import re

# A HuggingFace repo id is "namespace/name" (or just "name"): one optional slash,
# no other path separators, no leading slash, no spaces. Used to tell a repo id
# apart from a local path so we can give a clear error instead of the confusing
# ``HFValidationError: Repo id must be in the form ...`` raised for absolute paths.
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*(?:/[A-Za-z0-9][\w.-]*)?$")


def _resolve_model_source(value: str, label: str) -> str:
    """Return a model id/path that from_pretrained / ImageReward can consume.

    Resolution order:
      1. Existing local path  -> used verbatim.
      2. HuggingFace repo id  -> returned as-is for the hub to resolve/download.
      3. Anything else (e.g. an absolute path that does not exist) -> a clear
         FileNotFoundError, instead of letting transformers re-interpret the
         string as a repo id and crash with a cryptic HFValidationError.
    """
    if os.path.exists(value):
        return value
    if _REPO_ID_RE.match(value):
        return value  # repo id — let transformers / the hub handle it
    raise FileNotFoundError(
        f"{label} path does not exist on disk: {value!r}. "
        f"Pass a valid local directory (or a HuggingFace repo id such as "
        f"'openai/clip-vit-large-patch14') via --{label.lower().replace(' ', '_')} "
        f"or the EVAL_*_MODEL_PATH env var."
    )

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

    # Resolve model sources: local path (must exist) or HuggingFace repo id.
    clip_src = _resolve_model_source(clip_model_path, "CLIP model")
    imagereward_src = _resolve_model_source(imagereward_model_path, "ImageReward model")

    # Load models
    clip_model = CLIPModel.from_pretrained(clip_src)
    clip_model = clip_model.to(device)  # type: ignore
    clip_processor = CLIPProcessor.from_pretrained(clip_src)

    # Load ImageReward model
    if os.path.isdir(imagereward_src):
        med_config = os.path.join(imagereward_src, "med_config.json")
        imagereward_path = os.path.join(imagereward_src, "ImageReward.pt")
        imagereward_model = RM.load(imagereward_path, download_root=imagereward_src, med_config=med_config).to(device)
    else:
        # repo id, e.g. "zai-org/ImageReward" — let ImageReward download it.
        imagereward_model = RM.load(imagereward_src).to(device)
    
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
    parser.add_argument('--reference_folder', type=str, default=None, help='Reference images folder for quality metrics')
    parser.add_argument('--prompt_file', type=str, default="assets/prompts/DrawBench200.txt", help='Prompts file')
    # Local path or HuggingFace repo id. Override via CLI or the EVAL_*_MODEL_PATH env var.
    # Examples: /mnt/workspace/hkl/models/openai/clip-vit-large-patch14, laion/CLIP-ViT-g-14-laion2B-s12B-b42K
    parser.add_argument('--clip_model_path', type=str, default=DEFAULT_CLIP_MODEL_PATH)
    parser.add_argument('--imagereward_model_path', type=str, default=DEFAULT_IMAGEREWARD_MODEL_PATH)

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