# FasterCache SDXL 缓存策略说明

本目录下 `cache_functions/cal_type.py` 的步调度逻辑与 `taylorseer/` **完全对齐**，仅将变量名映射为 FasterCache 风格。

## 参数含义

| 参数 | 含义 | 对应 TaylorSeer 参数 |
|------|------|----------------------|
| `fc_start_step` | 温暖启动步数。所有 `step < fc_start_step` 的步都执行 **full**（完整计算），用于积累初始特征。 | `first_enhance` |
| `fc_interval` | 缓存周期长度。warm-up 结束后，每连续 `fc_interval` 步为一个周期：其中 **1 步 full**，其余 `fc_interval - 1` 步 **cache**（跳过计算）。 | `interval` |
| `fc_alpha` | 线性外推系数。在 cache 步中，预测特征为：`predicted = new + alpha * (new - old)`。 | —（FasterCache 特有） |

## 调度示例

以 `fc_start_step = 3, fc_interval = 6` 为例：

| step | 类型 | cache_counter | 说明 |
|------|------|---------------|------|
| 0 | full | 0 | warm-up（step < 3） |
| 1 | full | 0 | warm-up |
| 2 | full | 0 | warm-up |
| 3 | cache | 1 | 周期开始，第 1 个 cache 步 |
| 4 | cache | 2 | |
| 5 | cache | 3 | |
| 6 | cache | 4 | |
| 7 | cache | 5 | |
| 8 | full | 0 | cache_counter 达到 `fc_interval - 1 = 5`，重置为 full |
| 9 | cache | 1 | 新周期开始 |
| 10 | cache | 2 | |
| ... | ... | ... | ... |

## 与 TaylorSeer 的差异点

| 维度 | TaylorSeer | FasterCache（当前实现） |
|------|------------|-------------------------|
| **步调度** | `first_enhance` + `interval` + `cache_counter` | `fc_start_step` + `fc_interval` + `cache_counter`，**逻辑完全一致** |
| **特征外推** | 泰勒展开：`taylor_formula`（多阶导数多项式） | 线性外推：`cache_predict = new + alpha * (new - old)` |
| **缓存存储** | `derivative_approximation` 存储多阶导数 | `cache_store` 仅存储 `old` 和 `new` 两个张量 |

## 核心代码对照

### TaylorSeer (`taylorseer/cache_functions/cal_type.py`)
```python
first_step = (current['step'] < cache_dic['first_enhance'])
if (first_step) or (current['cache_counter'] == cache_dic['interval'] - 1):
    current['type'] = 'full'
    current['cache_counter'] = 0
else:
    current['type'] = 'cache'
    current['cache_counter'] += 1
```

### FasterCache (`fastercache_sdxl/cache_functions/cal_type.py`)
```python
first_step = (current['step'] < cache_dic['fc_start_step'])
if (first_step) or (current['cache_counter'] == cache_dic['fc_interval'] - 1):
    current['type'] = 'full'
    current['cache_counter'] = 0
else:
    current['type'] = 'cache'
    current['cache_counter'] += 1
```
