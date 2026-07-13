# FLUX 量化 + TaylorSeer 指南

> 基于[HuggingFace Issue 讨论](https://github.com/huggingface/diffusers/issues/9295)整理，并补充了本项目的实测结果。

## 1. Diffusers 兼容的量化方案

| 模型 | diffusers 兼容 | 实测状态 | 适用显存 | 说明 |
|------|---------------|---------|---------|------|
| `black-forest-labs/FLUX.1-dev` (BF16) | ✅ | 可用 | 35GB+ | 原版模型，无量化 |
| `black-forest-labs/FLUX.1-dev` (FP8) | ⚠️ | **失败** | — | `transformers` 加载 text encoder 时 `Float8_e4m3fnStorage` 报错 |
| `sayakpaul/flux.1-dev-nf4` | — | 未测 | 6-12GB | 社区预量化 NF4 |
| `lllyasviel/flux1-dev-bnb-nf4` | ⚠️ | **失败** | — | 单文件 `safetensors` 在 `dispatch_model` 阶段报 `Cannot copy out of meta tensor` |
| **原版 + 在线 NF4** | ✅ | **成功** | **~19GB** | **当前唯一可行的量化路径**，对原版 transformer 在线做 BnB NF4 |

### 显存构成实测（在线 NF4 量化后）

仅对 `transformer` 做 NF4 量化，T5/CLIP/VAE 保持 `bfloat16`：

| 组件 | BF16 显存 | NF4 后显存 | 是否量化 |
|------|----------|-----------|---------|
| Transformer | ~24 GB | ~8–9 GB | ✅ 已量化 |
| T5 Text Encoder | ~9–10 GB | ~9–10 GB | ❌ 未量化 |
| CLIP Text Encoder | ~1 GB | ~1 GB | ❌ 未量化 |
| VAE | ~0.3 GB | ~0.3 GB | ❌ 未量化 |
| 调度器 / 其他 | ~0.5 GB | ~0.5 GB | ❌ 未量化 |
| **总计** | **~35 GB** | **~19 GB** | — |

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

## 3. 实测结论

- **在线 NF4 量化是唯一在当前 diffusers + transformers 组合下实测通过的路径。**
- **FP8 失败原因**：PyTorch 不支持 `torch.float8_e4m3fn` 作为全局默认 dtype，`transformers` 在加载 text encoder（T5/CLIP）时调用 `torch.set_default_dtype(torch.float8_e4m3fn)`，触发 `TypeError: couldn't find storage object Float8_e4m3fnStorage`。
- **lllyasviel 单文件 NF4 失败原因**：`FluxTransformer2DModel.from_single_file()` 加载 BnB 量化权重后，模型权重处于 `meta` 设备；后续 `DiffusionPipeline.from_pretrained(..., transformer=transformer)` 在 `dispatch_model` → `model.to(device)` 时尝试拷贝 meta tensor，导致 `NotImplementedError: Cannot copy out of meta tensor; no data!`。

## 4. 适配 flux_diffusers 代码

当前 `get_torch_dtype()`（`inference_utils.py`）已支持 `float16/bfloat16/float32/float8`：

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

### 在线 NF4 量化（推荐）

```bash
cd flux_diffusers/taylorseer_flux
python diffusers_taylorseer_flux.py \
    --model /mnt/data0/pretrained_models/black-forest-labs/FLUX.1-dev \
    --steps 50 --dtype bfloat16 \
    --quantize nf4 \
    --guidance_scale 7.5 \
    --prompt "your prompt"
```

`batch_infer.py` 与 `scripts/infer_taylorseer_multi_flux.sh` 同样支持 `--quantize nf4`。

### FP8 用法（已知会失败，仅供参考）

```bash
python diffusers_taylorseer_flux.py \
    --model /path/to/FLUX.1-dev \
    --steps 50 --dtype float8 \
    --guidance_scale 7.5 \
    --prompt "your prompt"
```

> 在当前环境下会报 `Float8_e4m3fnStorage` 错误，暂不可用。

### BnB NF4 单文件加载（已知会失败）

```bash
python diffusers_taylorseer_flux.py \
    --model /mnt/data0/pretrained_models/black-forest-labs/FLUX.1-dev \
    --transformer_file /path/to/flux1-dev-bnb-nf4-v2.safetensors \
    --steps 50 --dtype bfloat16 \
    --prompt "your prompt"
```

> 会报 `Cannot copy out of meta tensor; no data!`，暂不可用。

## 5. 量化 + TaylorSeer 注意事项

**FP8 精度风险**：TaylorSeer 缓存中间激活做 Taylor 近似，FP8 精度低，缓存误差可能累积放大，导致生成质量下降。

建议对比流程：
1. `bfloat16` + TaylorSeer（baseline）
2. `float8` + TaylorSeer（实验组）
3. 比较输出图片质量（非纯黑/纯噪声即为通过）

如果 FP8 质量明显下降，可能需要让缓存部分保持更高精度（需改 `taylorseer_core/math.py` 的 cache 存储 dtype）。

**NF4 + TaylorSeer**：NF4 量化由 BnB 在推理时动态反量化到 bfloat16 计算，对 TaylorSeer 缓存的精度影响较小，兼容性优于 FP8。
