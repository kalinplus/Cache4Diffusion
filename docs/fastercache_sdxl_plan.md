# FasterCache for SDXL 实现计划

## 背景

当前仓库有三种 SDXL 加速方法：
- `stable_diffusion_xl/` — 无 cache 基线
- `taylorseer/` — Taylor 展开缓存所有子模块输出
- `freqca/` — 在 TaylorSeer 基础上增加频域分解 + Hermite + Z-cache

Qwen-Image 上还有 `qwen_fastercache/` 的 FasterCache 实现，但它是针对 DiT 架构的（只缓存 attention，用线性外推）。

本计划的目标是将 FasterCache 方法适配到 SDXL UNet，复用 taylorseer 的目录结构和 monkey-patching 基础设施，只替换缓存存储结构和预测公式。

## 目标

复制 `taylorseer/` 为新目录 `fastercache_sdxl/`，保留其 UNet monkey-patching 框架，将缓存策略从 Taylor 展开替换为 FasterCache 的线性外推，使三者的评测结果可横向对比。

## 步骤

### Step 1: 复制目录

```bash
cp -r taylorseer/ fastercache_sdxl/
rm -rf fastercache_sdxl/__pycache__ fastercache_sdxl/run.log fastercache_sdxl/samples/
```

### Step 2: 修改 `cache_functions/cache_init.py`

**改什么：**

将 `cache_dic` 的参数从 `interval, max_order, first_enhance` 改为 `fc_start_step, fc_interval, fc_alpha`：

```python
cache_dic['fc_start_step'] = kwargs['fc_start_step']  # int, e.g. 15
cache_dic['fc_interval']   = kwargs['fc_interval']    # int, e.g. 2
cache_dic['fc_alpha']      = kwargs['fc_alpha']       # float, e.g. 0.3
```

将 `current` 的状态变量简化：

```python
current['step'] = 0
# 去掉 cache_counter，直接用 step % interval 判断
```

cache 字典的嵌套初始化结构（`cache[-1]` 下的 downblocks/midblock/upblocks 层级）保持不变，因为 `pipe_with_cache` 和各子模块 forward 中通过 `current['module']`、`current['submodule']`、`current['idx']` 等键来索引的路径不变。

### Step 3: 修改 `cache_functions/cal_type.py`

**改什么：**

用 `fc_start_step` + 取模判断替换 `first_enhance` + `cache_counter` 逻辑：

```python
def cal_type(cache_dic, current):
    if current['step'] <= cache_dic['fc_start_step']:
        current['type'] = 'full'
    elif current['step'] % cache_dic['fc_interval'] == 0:
        current['type'] = 'full'
    else:
        current['type'] = 'cache'
```

去掉了 `current['cache_counter']`、`current['activated_steps']` 等 TaylorSeer 专用的状态变量。

### Step 4: 修改 `cache_functions/cache_utils.py`

**改什么：** 核心改动。删除 `derivative_approximation` 和 `taylor_formula`，替换为 `cache_store` 和 `cache_predict`。

```python
def cache_store(cache_entry, feature):
    """Full step 后调用：old <- new, new <- feature"""
    cache_entry["old"] = cache_entry.get("new")
    cache_entry["new"] = feature


def cache_predict(cache_entry, alpha):
    """Cache step 调用：线性外推恢复"""
    if cache_entry.get("old") is not None:
        return cache_entry["new"] + alpha * (cache_entry["new"] - cache_entry["old"])
    else:
        return cache_entry["new"]  # 第一次跳步，纯复用
```

`pipe_with_cache` 保持不变（它只做 monkey-patch，不涉及缓存逻辑）。

`__init__.py` 的导出改为：

```python
from .cache_init import cache_init
from .cache_utils import cache_store, cache_predict, pipe_with_cache
```

### Step 5: 修改 `models/attention.py` — BasicTransformerBlock.forward

**改什么：** 替换 `derivative_approximation` / `taylor_formula` 调用。

Full step 中，原来对 attn1、attn2、mlp 各调用一次 `derivative_approximation` 存 Taylor 因子，改为各调用一次 `cache_store`：

```python
# 原来（TaylorSeer）：
from cache_functions import derivative_approximation, taylor_formula
if current['type'] == 'full':
    ...正常计算 attn1...
    updated_taylor_factors = derivative_approximation(
        cache_dic=cache_dic['cache'][-1][...][current['subsubsubmodule']],
        current=current, max_order=cache_dic['max_order'], ...)
    cache_dic['cache'][-1][...][current['subsubsubmodule']] = updated_taylor_factors
else:
    attn_output = taylor_formula(cache_dic=cache_dic['cache'][-1][...][current['subsubsubmodule']], current=current)

# 改为（FasterCache）：
from cache_functions import cache_store, cache_predict
if current['type'] == 'full':
    ...正常计算 attn1...
    cache_store(cache_dic['cache'][-1][...][current['subsubsubmodule']], attn_output)
else:
    attn_output = cache_predict(cache_dic['cache'][-1][...][current['subsubsubmodule']], cache_dic['fc_alpha'])
```

attn1、attn2、mlp 三处同理。

### Step 6: 修改 `models/resnet.py` — ResnetBlock2D.forward

**改什么：** 同 Step 5 的模式，替换 `derivative_approximation` / `taylor_formula`。

Full step 中：`cache_store(..., hidden_states)`
Cache step 中：`hidden_states = cache_predict(..., cache_dic['fc_alpha'])`

residual 连接 `(input_tensor + hidden_states) / output_scale_factor` 保持不变。

### Step 7: 修改 `models/unets/unet_2d_blocks.py`

**改什么：** DownBlock2D、CrossAttnDownBlock2D、UNetMidBlock2DCrossAttn、CrossAttnUpBlock2D、UpBlock2D 各自 forward 中的 downsampler/upsampler cache 逻辑。

原来在 downsampler 处：

```python
if current['type'] == 'full':
    for downsampler in self.downsamplers:
        hidden_states = downsampler(hidden_states)
    updated_taylor_factors = derivative_approximation(...)
    cache_dic['cache'][-1][...] = updated_taylor_factors
else:
    hidden_states = taylor_formula(...)
```

改为：

```python
if current['type'] == 'full':
    for downsampler in self.downsamplers:
        hidden_states = downsampler(hidden_states)
    cache_store(cache_dic['cache'][-1][...], hidden_states)
else:
    hidden_states = cache_predict(cache_dic['cache'][-1][...], cache_dic['fc_alpha'])
```

每个 block 的 resnet 和 attention 内部的调用已经在 Step 5/6 中处理过了。

### Step 8: 修改 `sample.py`

**改什么：** 命令行参数。

去掉 `--interval`、`--max_order`、`--first_enhance`，替换为：

```python
parser.add_argument('--fc_start_step', type=int, default=15)
parser.add_argument('--fc_interval', type=int, default=2)
parser.add_argument('--fc_alpha', type=float, default=0.3)
```

`SamplingOptions` dataclass 和传给 `cache_init` 的 kwargs 对应更新。

### Step 9: 修改 `run.sh`

**改什么：** 替换为 FasterCache 的参数 sweep。例如：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 sample.py \
    --fc_start_step 15 --fc_interval 2 --fc_alpha 0.0 --output_dir samples/reuse_i2_a0
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 sample.py \
    --fc_start_step 15 --fc_interval 2 --fc_alpha 0.3 --output_dir samples/linear_i2_a0.3
# ...
```

### Step 10: 删除 `cache_functions/cal_type.py`

`cal_type` 的逻辑已内联到各子模块的判断中（通过 `current['type']`），或保留但简化为 Step 3 的版本。建议保留，因为 `pipeline_stable_diffusion_xl.py` 的 denoising loop 中显式调用了 `cal_type()`。

## 不改什么

| 文件/内容 | 原因 |
|-----------|------|
| `models/unets/unet_2d_condition.py` | UNet forward 只传递 `cache_dic`/`current`，不涉及缓存逻辑 |
| `models/transformers/transformer_2d.py` | Transformer2DModel forward 只设置 `current['subidx']` 后透传，不涉及缓存逻辑 |
| `pipelines/pipeline_stable_diffusion_xl.py` | denoising loop 中调 `cal_type()` 和传 `cache_dic`/`current`，接口不变 |
| `cache_functions/cache_init.py` 中 cache 的嵌套初始化 | `cache[-1]` 下的 downblocks/midblock/upblocks 层级结构和 `pipe_with_cache` 一致，改了反而要动 models |
| `models/attention.py` 中的 `BasicTransformerBlock.__init__` | 网络结构不变 |
| `models/resnet.py` 中除 `ResnetBlock2D.forward` 外的部分 | 网络结构不变 |
| `evaluate.py` | 评测逻辑不变 |
| `prompts/DrawBench200.txt` | 不变 |

## 风格参考

- 缓存存储的 key 命名：`"old"` / `"new"`，与 `qwen_fastercache/transformer_qwenimage.py` 中 `QwenImageTransformerBlock._fc_caches` 的风格一致
- 预测公式：`new + alpha * (new - old)`，与 `qwen_fastercache` 第 404-411 行一致
- 参数命名：`fc_start_step`、`fc_interval`、`fc_alpha`，与 `qwen_fastercache/sample_ddp.py` 的 `--fc_start_step`、`--fc_interval`、`--fc_alpha` 一致
- 目录名 `fastercache_sdxl/`，与 `qwen_fastercache/` 命名风格一致

## 如何验证

1. **单图生成测试**：用单张 prompt 跑一遍 `sample.py`，确认无报错，输出图像非全黑/全噪声
2. **FLOPs 对比**：`--test_FLOPs` 跑一次，打印的 FLOPs 应低于基线，加速比 ≈ `(N - start_step) / N * (interval - 1) / interval`（粗估，实际取决于 interval 和 warm-up）
3. **质量对比**：用 `evaluate.py` 对比 ClipScore、ImageReward、PickScore，FasterCache (alpha=0.3) 应与 TaylorSeer (interval=2) 接近
4. **alpha 消融**：`alpha=0.0`（纯复用）vs `alpha=0.3`（线性外推）vs `alpha=0.5`（更大外推），观察质量-速度 trade-off
5. **interval 消融**：`interval=2` vs `3` vs `4`，验证间隔越大加速越明显但质量下降越严重
