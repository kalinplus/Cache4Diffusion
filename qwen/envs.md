
Prepare enviroment
```bash
# create environment 
conda create -n qwen python=3.10
conda activate qwen
pip install transformers==4.55.4 peft
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/huggingface/diffusers

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
hf download Qwen/Qwen-Image
hf download Qwen/Qwen-Image-Edit
hf download lightx2v/Qwen-Image-Lightning

# evaluate
hf download zai-org/ImageReward
hf download laion/CLIP-ViT-g-14-laion2B-s12B-b42K

# GEdit
hf download stepfun-ai/GEdit-Bench --repo-type=dataset
hf download Qwen/Qwen2.5-VL-72B-Instruct-AWQ
```
