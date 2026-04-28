# TaylorSeer Smooth 算法实现记录（SDXL UNet）

参考文档：`/home/hkl/Cache4Diffusion/docs/TaylorSeer_Smooth_Algorithm.md`

---

## 实现概要

### 修改文件清单

| 文件 | 改动 |
|------|------|
| `cache_functions/cache_init.py` | 提取 `_create_block_cache()`，初始化 `cache[-2]` 镜像结构，添加平滑配置项 |
| `cache_functions/cache_utils.py` | 新增 `_get_module_cache`、平滑函数、`update_cache_or_approximate` 统一入口 |
| `models/resnet.py` | `derivative_approximation` + `taylor_formula` → `update_cache_or_approximate` |
| `models/attention.py` | `attn1/attn2/mlp` 三处调用 → `update_cache_or_approximate` |
| `cache_functions/__init__.py` | 导出 `update_cache_or_approximate` |
| `sample.py` | 新增 `--use_smoothing`、`--use_hybrid_smoothing`、`--smoothing_method`、`--smoothing_alpha` |

### 使用方式

```bash
# 无平滑（与原始 TaylorSeer 完全等价）
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 sample.py --interval 6

# 指数平滑
TS_DEBUG_SMOOTH=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 sample.py --interval 6 --use_smoothing --smoothing_alpha 0.75

# 混合平滑（一阶用原始特征，二阶+用平滑特征）
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 sample.py --interval 6 --use_smoothing --use_hybrid_smoothing

# 移动平均
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 sample.py --interval 6 --use_smoothing --smoothing_method moving_average
```

调试输出通过环境变量 `TS_DEBUG_SMOOTH=1` 控制，每个 timestep 仅打印首条模块信息，避免刷屏。

---

## 架构差异：SDXL UNet vs Flux/Qwen DiT

参考文档和已有实现（`/home/hkl/Cache4Diffusion/qwen/taylorseer/`、`/home/hkl/Cache4Diffusion/flux_simple/`）均面向 DiT 架构（Transformer blocks），其缓存路径是扁平的 `cache[-1][stream][layer][module]`，而 SDXL UNet 的缓存路径是多层嵌套的：

```
cache[-1][module][submodule][subsubmodule][idx][subidx][subsubsubmodule]
         │        │           │           │      │       │
      downblocks  DownBlock2D_0  resnet     0     -       -
      midblock    UNetMidBlock2DCrossAttn  attention  0  0  attn1/mlp/attn2
      upblocks    CrossAttnUpBlock2D_0     upsampler
```

路径中的键并非全部存在，取决于 `subsubmodule` 类型：

| `subsubmodule` | 有 `idx`? | 有 `subidx`? | 有 `subsubsubmodule`? |
|---------------|-----------|-------------|---------------------|
| `resnet` | 是 | 否 | 否 |
| `attention` | 是 | 是 | 是 |
| `downsampler` / `upsampler` | 否 | 否 | 否 |

---

## 遇到的 Bug 及修复

### Bug 1：`_get_module_cache` 残留键导致 KeyError

**症状**：

```
File "cache_utils.py", line 14, in _get_module_cache
    d = d[current['subidx']]
KeyError: 9
```

`current` 是贯穿整个 UNet forward 的共享可变字典。`CrossAttnDownBlock2D` 的 attention 调用会设置 `current['subidx']`（值为 transformer block 索引），处理完成后该键仍残留在 `current` 中。随后进入 `UNetMidBlock2DCrossAttn` 调用第一个 resnet 时，`_get_module_cache` 通过 `if 'subidx' in current` 拾取了残留值，尝试用 step 值作为字典索引，触发 KeyError。

**根本原因**：原始代码中 resnet 直接用固定路径 `cache_dic['cache'][-1][...][current['idx']]` 访问，不会检查 `subidx`；而 `_get_module_cache` 无条件检查了所有可能的键。

DiT 架构不存在此问题，因为每个 transformer block 的 forward 都会完整设置 `current` 的所有键，不存在 resnet-only 的调用路径。

**修复**：让 `_get_module_cache` 根据 `subsubmodule` 类型决定是否访问深层键：

```python
def _get_module_cache(cache_dic, current, history_idx=-1):
    d = cache_dic['cache'][history_idx]
    d = d[current['module']][current['submodule']][current['subsubmodule']]
    if current['subsubmodule'] in ('resnet', 'attention') and 'idx' in current:
        d = d[current['idx']]
    if current['subsubmodule'] == 'attention' and 'subidx' in current:
        d = d[current['subidx']]
    if current['subsubmodule'] == 'attention' and 'subsubsubmodule' in current:
        d = d[current['subsubsubmodule']]
    return d
```

**迁移启示**：UNet 架构中 `current` 字典在不同 block 类型（resnet / attention / downsampler）之间共享，各 block 的 forward 只设置自己需要的键，不清理其他 block 的键。在编写路径解析逻辑时，必须基于 `subsubmodule` 类型做条件判断，而非盲目检查所有键是否存在。

---

### Bug 2（文档已知 Bug 3 的体现）：历史保存时序

参考文档详细记录了此 Bug（`shift_cache_history` 在 `update_cache_or_approximate` 开头调用导致 `cache[-2]` 被提前覆盖），SDXL 实现中直接规避了此问题：

**做法**：不使用单独的 `shift_cache_history` 函数，而是在 `update_cache_or_approximate` 的 `full` 分支内部，**先保存历史再计算导数**：

```python
cache_prev.clear()
cache_prev.update(dict(cache_now))   # 浅拷贝保存历史
# ... 然后计算导数并更新 cache_now ...
```

**注意 `dict()` 浅拷贝的必要性**：如果写 `cache_prev.update(cache_now)` 而不加 `dict()`，`cache_prev` 和 `cache_now` 会持有相同的内部字典引用，后续 `cache_now.clear()` 会同时清空 `cache_prev`。这在文档 Bug 1 中有详细分析。

---

## 关键设计决策

| 决策 | 说明 |
|------|------|
| `_get_module_cache` 按 `subsubmodule` 类型条件解析路径 | 适配 UNet 混合 block 类型（resnet / attention / sampler）共享 `current` 字典的特性 |
| 保留原有 `derivative_approximation` / `taylor_formula` 签名 | 局部缓存字典传参方式在 SDXL 代码中大量使用；通过新增统一入口最小化改动 |
| 历史保存内联到 `update_cache_or_approximate` | 避免文档 Bug 3 的时序问题 |
| `cache[-2]` 通过 `_create_block_cache()` 函数式生成 | 与 `cache[-1]` 结构完全一致，避免手动复制出错 |
| 调试输出限制为每 step 首条模块 | UNet 有数十个缓存模块，不加限制会产生巨量日志 |
