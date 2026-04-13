
Prepare enviroment
```bash
# create environment 
conda create -n stablediffusion python=3.10
conda activate stablediffusion
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install diffusers --upgrade
pip install invisible_watermark transformers accelerate safetensors

# evaluate
pip install opencv-python lpips scikit-image image-reward
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/chengzegang/calculate-flops.pytorch.git
pip install transformers==4.55.4
```

Set environment variables (in `.bashrc` file)
```bash
export XDG_CACHE_HOME="/path/to/.cache"
export HF_ENDPOINT="https://hf-mirror.com"
```

Download models and dataset
```bash
# model
hf download stabilityai/stable-diffusion-xl-base-1.0

# evaluate
hf download zai-org/ImageReward
hf download laion/CLIP-ViT-g-14-laion2B-s12B-b42K
hf download laion/CLIP-ViT-H-14-laion2B-s32B-b79K
hf download yuvalkirstain/PickScore_v1
```

Official code
```python
from diffusers import DiffusionPipeline
import torch

# load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda")
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

prompt = "A majestic lion jumping from a big stone at night"

# run both experts
image = base(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images
image = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
).images[0]
```