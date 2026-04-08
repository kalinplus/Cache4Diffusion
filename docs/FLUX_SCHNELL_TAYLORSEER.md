# FLUX-schnell + TaylorSeer 使用指南

## 概述

FLUX-schnell 是 Black Forest Labs 发布的快速文生图模型，仅需 4 步即可生成高质量图片。本项目在 `flux/taylorseer/` 下集成了 TaylorSeer 缓存加速方案，可跳过部分 transformer 子模块计算以进一步提速。

## 涉及文件

| 文件 | 作用 |
|------|------|
| `flux/taylorseer/src/sample.py` | 批量推理入口（单卡） |
| `flux/taylorseer/src/sample_ddp.py` | 多卡 DDP 推理 |
| `flux/taylorseer/src/flux/sampling.py` | 去噪循环、时间步调度 |
| `flux/taylorseer/src/flux/model.py` | Flux Transformer，`FluxParams` 中 `guidance_embed=False` 是 schnell 的唯一结构差异 |
| `flux/taylorseer/src/flux/util.py` | 模型路径配置，定义 schnell 的 `ModelSpec` |
| `flux/taylorseer/src/flux/modules/layers.py` | DoubleStreamBlock / SingleStreamBlock，含缓存钩子 |
| `flux/taylorseer/src/flux/modules/cache_functions/` | TaylorSeer 缓存初始化、Taylor 公式、full/cache 决策 |
| `flux/taylorseer/sample_schnell.sh` | 运行脚本（批量参数网格搜索） |

同一套代码也被复制到 `flux/flux/`、`flux/freqca/`、`flux/toca/`、`flux/duca/`、`flux/teacache/` 等目录（各方法仅有缓存逻辑不同）。

## 输入输出

### 输入

- `--prompt_file`：每行一条 prompt 的文本文件（默认 `DrawBench200.txt`）
- `--width` / `--height`：图片尺寸（默认 1024x1024，须为 16 的倍数）
- `--num_steps`：**schnell 固定为 4 步**（代码 assert 强制）
- `--guidance`：传递给模型但**被完全忽略**（`guidance_embed=False`）
- `--seed`、`--batch_size`、`--num_images_per_prompt`
- TaylorSeer 参数：`--interval`、`--max_order`、`--first_enhance`

### 输出

- JPEG 图片保存到 `{output_dir}/img_{idx}.jpg`（quality=95）
- EXIF 元数据包含 model 名称，可选嵌入 prompt

## 模型加载

四个组件：

1. **T5** (`google/t5-v1_1-xxl`)：schnell 用 `max_length=256`（dev 用 512）
2. **CLIP** (`openai/clip-vit-large-patch14`)：`max_length=77`，所有模型相同
3. **Flux Transformer**：从 safetensors 加载
4. **VAE Autoencoder**：`ae.safetensors`

模型路径配置（`sample_schnell.sh` 中已设置）：

```bash
export FLUX_MODEL="/mnt/data0/pretrained_models/black-forest-labs/FLUX.1-schnell/flux1-schnell.safetensors"
export FLUX_AE="/mnt/data0/pretrained_models/black-forest-labs/FLUX.1-schnell/ae.safetensors"
export T5_MODEL_PATH="/mnt/data0/pretrained_models/google/t5-v1_1-xxl"
export CLIP_MODEL_PATH="/mnt/data0/pretrained_models/openai/clip-vit-large-patch14"
```

## 调用方法

```bash
# 单次运行
CUDA_VISIBLE_DEVICES=0 python flux/taylorseer/src/sample.py \
    --prompt_file prompts/DrawBench200.txt \
    --model_name flux-schnell \
    --num_steps 4 \
    --seed 0 \
    --interval 4 \
    --max_order 0 \
    --first_enhance 1 \
    --output_dir samples/test

# 批量参数搜索（使用 sample_schnell.sh）
bash flux/taylorseer/sample_schnell.sh

# 多卡 DDP
torchrun --nproc_per_node=N flux/taylorseer/src/sample_ddp.py --model_name flux-schnell ...
```

### 环境变量控制缓存行为

| 变量 | 作用 | 默认值 |
|------|------|--------|
| `USE_SMOOTHING` | 启用平滑 | `False` |
| `USE_HYBRID_SMOOTHING` | 混合平滑模式 | `False` |
| `SMOOTHING_METHOD` | `exponential` 或 `moving_average` | `exponential` |
| `SMOOTHING_ALPHA` | 平滑系数 | `0.8` |

## 生图流程

```
1. 加载 T5 (maxlen=256) + CLIP → 编码 prompt
2. 生成随机噪声 img ∈ ℝ^(B, 16, H/16, W/16)
3. 构建时间步：linspace(1, 0, 5) = [1.0, 0.8, 0.6, 0.4, 0.0]（线性，无 shift）
4. 循环 4 步：
   for t_curr, t_prev in [(1.0,0.8), (0.8,0.6), (0.6,0.4), (0.4,0.0)]:
     ┌─ Flux.forward():
     │  ├─ 线性投影: img, txt, vec(含 CLIP pooler + timestep embed)
     │  ├─ RoPE 位置编码
     │  ├─ cal_type() 决定本步是 full 还是 cache
     │  ├─ 19 个 DoubleStreamBlock (text+image 双流):
     │  │   full → 正常计算并缓存 img_attn/img_mlp/txt_attn/txt_mlp
     │  │   cache → 用 Taylor 公式近似替代（跳过 attention+MLP）
     │  ├─ 拼接 text+image → 38 个 SingleStreamBlock
     │  └─ 分离 text tokens → FinalLayer
     └─ img = img + (t_prev - t_curr) * pred
5. VAE 解码: latent → pixel
6. clamp(-1,1) → 加水印 → 保存 JPEG
```

## Schnell vs Dev 关键差异

| 特性 | schnell | dev |
|------|---------|-----|
| 步数 | 固定 4 | 默认 50 |
| guidance | 忽略（`guidance_embed=False`） | 通过 MLP 嵌入 |
| T5 max_length | 256 | 512 |
| 时间步调度 | 线性（`shift=False`） | 非线性 shift |
| TRT | 移除 guidance 输入 | 保留 |

## TaylorSeer 缓存与 Schnell 的结合

默认参数下（`interval=4, max_order=0, first_enhance=1`）：

- **Step 0**：Full 计算（`first_enhance` 强制）
- **Steps 1-3**：Cache 模式，用 Taylor 零阶近似（`max_order=0`），直接复用 step 0 的特征

75% 的 transformer 计算被跳过（19 个 DoubleBlock x 4 子模块 + 38 个 SingleBlock x 1 子模块）。由于 schnell 只有 4 步且忽略 guidance，仅 timestep embedding 变化，激进缓存策略效果较好。

### 参数建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `interval` | 2 或 4 | `interval=4` 仅 step 0 full；`interval=2` 在 step 0 和 2 各做一次 full |
| `max_order` | 0 或 1 | 0 = FORA 零阶（最快）；1 = 一阶（更精确） |
| `first_enhance` | 1 | schnell 只有 4 步，建议不超过 2 |

## 模型加载日志说明

运行时会看到以下 UNEXPECTED 警告，属于正常现象：

- **T5** `lm_head.weight UNEXPECTED`：T5 仅作为 text encoder 使用，不需要语言模型解码头
- **CLIP** 大量 `vision_model.* UNEXPECTED`：加载的是 `CLIPTextModel`（纯文本编码器），而 checkpoint 包含完整 CLIP（text + vision encoder + 对比学习头），视觉部分正常丢弃

这些警告不影响输出质量。
