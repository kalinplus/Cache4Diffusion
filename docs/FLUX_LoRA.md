# FLUX LoRA 使用指南

> 原文：[Using Flux.1 Diffusion Model with Python — #3 — Using Flux LoRAs](https://www.fluxai.cn/detail/using-flux1-diffusion-model-with-python3using-20240920)

## 1. LoRA 格式

通过 `safetensors` 的 key 前缀可以识别 LoRA 格式，共三种：

| 格式 | key 前缀 | 自动转换 |
|------|----------|----------|
| Diffusers | `transformer.*` | 原生支持 |
| xlabs | `diffusion_model.double_blocks.*` 或 `double_blocks.*` | 新版 diffusers 自动转换 |
| kohya | `lora_unet.*` | 新版 diffusers 自动转换 |

检测方法：

```python
from safetensors.torch import load_file
original_dict = load_file("/path/to/lora.safetensors")
print(original_dict.keys())
```

新版 diffusers（需包含 `src/diffusers/loaders/lora_conversion_utils.py` 中的 `_convert_kohya_flux_lora_to_diffusers` 和 `_convert_xlabs_flux_lora_to_diffusers`）可自动将三种格式统一转换为 Diffusers 格式，无需手动转换。

## 2. 环境准备

```bash
pip install -U diffusers
pip install git+https://github.com/xhinker/sd_embed.git@main  # 可选，用于无限长提示
```

## 3. 核心代码

```python
from diffusers import DiffusionPipeline, FluxTransformer2DModel
import torch

# 加载模型
model_path = "/path/to/FLUX.1-dev"
transformer = FluxTransformer2DModel.from_pretrained(
    model_path, subfolder="transformer", torch_dtype=torch.bfloat16
)
pipe = DiffusionPipeline.from_pretrained(
    model_path, transformer=transformer, torch_dtype=torch.bfloat16
)

# 加载 LoRA（自动识别格式）
pipe.load_lora_weights("/path/to/lora.safetensors")
pipe.enable_model_cpu_offload(gpu_id=0)

# 推理
image = pipe(
    prompt="A close-up portrait with golden art face",
    width=1680, height=1024,
    num_inference_steps=30,
    guidance_scale=3.5,
    joint_attention_kwargs={"scale": 1.0},
    generator=torch.Generator().manual_seed(42),
).images[0]
```

多个 LoRA 可通过 adapter 机制组合：

```python
pipe.load_lora_weights("lora1.safetensors", adapter_name="base")
pipe.load_lora_weights("lora2.safetensors", adapter_name="style")
pipe.set_adapters(["base", "style"], weights=[1.0, 0.8])
```

## 4. LoRA + TaylorSeer 集成

### 使用 flux_diffusers 版本（推荐）

项目中有两个 Flux + TaylorSeer 实现，推荐使用 `flux_diffusers/taylorseer_flux`：

| 方面 | `flux_diffusers/taylorseer_flux` | `flux/taylorseer` |
|------|-------------------------------|-------------------|
| Pipeline | 标准 `DiffusionPipeline` | 自定义 `Flux` 类，手动加载各组件 |
| LoRA | `pipeline.load_lora_weights()` 一行搞定 | 自定义 `FluxLoraWrapper` + `LinearLora`，需理解内部实现 |
| Forward | Monkey-patch diffusers transformer forward | 缓存逻辑内置在自定义 model 里 |
| 兼容性 | 支持所有 diffusers/PEFT 生态 LoRA | 仅支持自定义 LoRA 格式 |

`flux_diffusers` 使用标准 diffusers pipeline，LoRA 接入只需在 patch forward 之前加一行：

```python
from diffusers import DiffusionPipeline
from model_adapters import patch_model_with_cache

pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipeline.load_lora_weights("/path/to/lora.safetensors")  # ← LoRA 加在这里
pipeline.transformer.__class__.num_steps = steps
patch_model_with_cache(pipeline.transformer, strategy_name="taylorseer", model_name="flux")
pipeline.to(device)
```

`flux/taylorseer` 的模型是完全自定义实现（`src/flux/model.py`），LoRA 也是自定义实现（`src/flux/modules/lora.py`），集成更复杂且不兼容 HuggingFace 标准 LoRA 生态。

## 参考文献

- [转换 Flux LoRA](https://github.com/kohya-ss/sd-scripts/blob/a61cf73a5cb5209c3f4d1a3688dd276a4dfd1ecb/networks/convert_flux_lora.py)
- [支持 kohya 和 xlabs loras for flux](https://github.com/huggingface/diffusers/pull/9295)

# 去哪找 LoRA
要找的 FLUX.1-dev 兼容 LoRA，同时它还要兼容 diffusers 库生态，这样才能一行加载搞定。
似乎可以从：https://huggingface.co/spaces/multimodalart/flux-lora-the-explorer 中找到。只要确定搜索到的库的标签里包括 Diffuseres