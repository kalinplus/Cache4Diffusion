
Prepare enviroment
```bash
# create environment 
conda create -n flux python=3.10
conda activate flux
cd flux
pip install .
pip install transformers==4.55.4 peft

# evaluate
pip install opencv-python lpips scikit-image image-reward
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/chengzegang/calculate-flops.pytorch.git

# GEdit
pip install megfile pandas datasets google-generativeai python-dotenv qwen_vl_utils autoawq
```

Set environment variables (in `.bashrc` file)
```bash
export XDG_CACHE_HOME="/data/public/.cache"
export HF_ENDPOINT="https://hf-mirror.com"
```

Download models and dataset
```bash
hf download --token YOUR_HFTOKEN black-forest-labs/FLUX.1-schnell
hf download --token YOUR_HFTOKEN black-forest-labs/FLUX.1-dev
hf download --token YOUR_HFTOKEN black-forest-labs/FLUX.1-Kontext-dev
hf download --token YOUR_HFTOKEN black-forest-labs/FLUX.1-Fill-dev

hf download google/t5-v1_1-xxl
hf download openai/clip-vit-large-patch14

# evaluate
hf download zai-org/ImageReward
hf download laion/CLIP-ViT-g-14-laion2B-s12B-b42K

# GEdit
hf download stepfun-ai/GEdit-Bench --repo-type=dataset
hf download Qwen/Qwen2.5-VL-72B-Instruct-AWQ
```
