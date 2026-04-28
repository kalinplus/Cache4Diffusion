# 如何启动 flash attention
Taylorseer/Taylorseer-HunyuanVideo 默认使用 flash attention 无需设置。不同于 Cache4Diffusion 中因为 Diffusers 库的要求需要额外指定

# 部署踩坑记录

## 环境配置：切勿安装 xfuser

**教训**：`pip install xfuser==0.4.0` 会将 PyTorch 从 **2.4.0** 强制升级到 **2.10.0**，导致 torchvision 报错 `RuntimeError: operator torchvision::nms does not exist`，整个环境不可用。

**原因**：xfuser 0.4.0 的依赖树要求高版本 PyTorch，conda 安装的 torch 2.4.0 会被 pip 覆盖。

**结论**：单卡推理不需要 xfuser，**不要安装**。如果确实需要多卡并行，需要单独创建环境并在安装后手动降级 torch，代价较大。

正确的环境配置只需 5 步：

```bash
# 1. 创建环境
conda create -n HunyuanVideo python==3.10.9

# 2. 激活环境
conda activate HunyuanVideo

# 3. 安装 PyTorch（CUDA 12.4）
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# 4. 安装 Python 依赖
python -m pip install -r requirements.txt

# 5. 安装 FlashAttention v2（必须加 --no-build-isolation，否则构建隔离环境里找不到 torch）
python -m pip install ninja
python -m pip install --no-build-isolation git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3
```

安装完成后验证：

```bash
python -c "import torch; print(torch.__version__)"  # 必须是 2.4.0
```

---

# 模型路径配置说明

## 问题背景

项目中存在两套独立的模型路径机制，当模型存放在非默认路径（如 `/mnt/data0/tencent/HunyuanVideo`）时，必须同时配置两者，否则会因路径找不到而报错。

## 两套路径机制

### 1. 命令行参数 `--model-base` 和 `--dit-weight`

| 参数 | 默认值 | 作用范围 |
|------|--------|---------|
| `--model-base` | `ckpts` | 仅用于验证目录是否存在，以及作为 `dit-weight` 的基准路径（但 `dit-weight` 默认值是硬编码绝对拼接，实际不会自动相对解析） |
| `--dit-weight` | `ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt` | 扩散 transformer 模型权重路径，作为绝对路径使用 |

这两个参数在 `hyvideo/config.py` 中定义，仅影响扩散模型权重的加载。

### 2. 环境变量 `MODEL_BASE`

定义在 `hyvideo/constants.py:67`：

```python
MODEL_BASE = os.getenv("MODEL_BASE", "./ckpts")
```

该变量用于拼接以下路径：

| 常量 | 拼接结果 | 用途 |
|------|---------|------|
| `VAE_PATH["884-16c-hy"]` | `{MODEL_BASE}/hunyuan-video-t2v-720p/vae` | 3D VAE 模型 |
| `TEXT_ENCODER_PATH["llm"]` | `{MODEL_BASE}/text_encoder` | MLLM 文本编码器 (llava-llama-3-8b) |
| `TEXT_ENCODER_PATH["clipL"]` | `{MODEL_BASE}/text_encoder_2` | CLIP 文本编码器 |
| `TOKENIZER_PATH["llm"]` | `{MODEL_BASE}/text_encoder` | MLLM tokenizer |
| `TOKENIZER_PATH["clipL"]` | `{MODEL_BASE}/text_encoder_2` | CLIP tokenizer |

当 `load_vae()` 和 `TextEncoder()` 未显式传入路径时，会回退使用这些默认值。

## 必要修改

当模型存放于自定义路径时（如 `/mnt/data0/tencent/HunyuanVideo`），需要在运行脚本中同时做以下修改：

### 设置环境变量

```bash
export MODEL_BASE=/mnt/data0/tencent/HunyuanVideo
```

使 `constants.py` 中的 VAE、text_encoder、tokenizer 路径全部指向正确位置。

### 添加命令行参数

```bash
--model-base /mnt/data0/tencent/HunyuanVideo \
--dit-weight /mnt/data0/tencent/HunyuanVideo/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
```

使 `sample_video.py` 能正确验证模型目录并加载扩散模型权重。

## 模型目录结构要求

自定义路径下的目录结构需与默认布局一致：

```
{MODEL_BASE}/
├── hunyuan-video-t2v-720p/
│   ├── transformers/
│   │   └── mp_rank_00_model_states.pt
│   └── vae/
│       ├── config.json
│       └── pytorch_model.pt
├── text_encoder/          # llava-llama-3-8b-v1_1-transformers
└── text_encoder_2/        # clip-vit-large-patch14
```

## 完整脚本示例

```bash
#!/bin/bash
export MODEL_BASE=/mnt/data0/tencent/HunyuanVideo

python3 sample_video.py \
    --model-base /mnt/data0/tencent/HunyuanVideo \
    --dit-weight /mnt/data0/tencent/HunyuanVideo/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --video-size 480 640 \
    --video-length 65 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --seed 42 \
    --flow-reverse \
    --use-cpu-offload \
    --save-path ./results
```
