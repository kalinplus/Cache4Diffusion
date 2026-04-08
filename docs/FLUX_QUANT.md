# FLUX 量化 + TaylorSeer 指南

> 基于[HuggingFace Issue 讨论](https://github.com/huggingface/diffusers/issues/9295)整理。

## 1. Diffusers 兼容的量化方案

| 模型 | diffusers 兼容 | 加载方式 | 适用显存 |
|------|---------------|----------|---------|
| `black-forest-labs/FLUX.1-dev` | ✅ | `torch_dtype=torch.bfloat16` | 16GB+ |
| `black-forest-labs/FLUX.1-dev` (FP8) | ✅ | `torch_dtype=torch.float8_e4m3fn` | 12GB+ |
| `sayakpaul/flux.1-dev-nf4` | ✅ | NF4 量化，`torch_dtype=torch.bfloat16` | 6-12GB |
| `lllyasviel/flux1-dev-bnb-nf4` | ✅ | BnB NF4 量化 | 6-12GB |
| `Kijai/flux-fp8` | ❌ | 仅 ComfyUI，结构不同 | — |

## 2. 环境要求

```bash
# PyTorch 2.4.0+（FP8 支持）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# diffusers / transformers 开发版
pip install git+https://github.com/huggingface/diffusers
pip install git+https://github.com/huggingface/transformers
pip install git+https://github.com/huggingface/accelerate

# transformers >= 4.43.2
pip install "transformers>=4.43.2"
```

## 3. 适配 flux_simple 代码

当前 `get_torch_dtype()`（`inference_utils.py`）仅支持 `float16/bfloat16/float32`，需增加 FP8 映射：

```python
# inference_utils.py — get_torch_dtype()
def get_torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float8":
        return torch.float8_e4m3fn
    return torch.float32
```

之后启动时传 `--dtype float8` 即可，`setup_pipeline()` 内部无需改动，因为 `DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float8_e4m3fn)` 是 diffusers 原生支持的。

### FP8 用法

```bash
cd flux_simple/taylorseer_flux
python diffusers_taylorseer_flux.py \
    --model /path/to/FLUX.1-dev \
    --steps 50 --dtype float8 \
    --guidance_scale 7.5 \
    --prompt "your prompt"
```

### NF4 用法

NF4 版本（如 `sayakpaul/flux.1-dev-nf4`）已经是量化好的权重，加载时仍用 `bfloat16`：

```bash
python diffusers_taylorseer_flux.py \
    --model sayakpaul/flux.1-dev-nf4 \
    --steps 50 --dtype bfloat16 \
    --prompt "your prompt"
```

## 4. 量化 + TaylorSeer 注意事项

**FP8 精度风险**：TaylorSeer 缓存中间激活做 Taylor 近似，FP8 精度低，缓存误差可能累积放大，导致生成质量下降。

建议对比流程：
1. `bfloat16` + TaylorSeer（baseline）
2. `float8` + TaylorSeer（实验组）
3. 比较输出图片质量（非纯黑/纯噪声即为通过）

如果 FP8 质量明显下降，可能需要让缓存部分保持更高精度（需改 `taylorseer_core/math.py` 的 cache 存储 dtype）。

**NF4 + TaylorSeer**：NF4 量化由 BnB 在推理时动态反量化到 bfloat16 计算，对 TaylorSeer 缓存的精度影响较小，兼容性应优于 FP8。
