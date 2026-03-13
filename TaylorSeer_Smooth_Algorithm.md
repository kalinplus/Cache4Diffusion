# TaylorSeer Smooth 算法文档

## 目录
1. [算法概述](#算法概述)
2. [算法作用](#算法作用)
3. [核心原理](#核心原理)
4. [实现细节](#实现细节)
5. [关键代码修改](#关键代码修改)
6. [配置参数](#配置参数)
7. [迁移指南](#迁移指南)

---

## 算法概述

TaylorSeer Smooth 是 TaylorSeer 缓存加速框架的可选增强功能，通过对缓存的特征值进行时间平滑处理，减少相邻时间步之间的突变，提高 Taylor 级数展开的近似精度。

**核心思想**：
- 在计算导数近似时，不直接使用原始特征值 `F_0, F_{-1}, F_{-2}`
- 而是先对历史特征序列进行平滑处理，得到 `F'_0, F'_{-1}, F'_{-2}`
- 再基于平滑后的特征计算导数，从而获得更稳定的 Taylor 展开系数

**支持两种平滑模式**：
1. **全局平滑（Global Smoothing）**：对所有阶导数都使用平滑后的特征计算
2. **混合平滑（Hybrid Smoothing）**：一阶导数使用原始特征，二阶及以上导数使用平滑特征

---

## 算法作用

### 1. 减少时间不连续性
扩散模型在相邻时间步之间的特征变化可能存在突变，直接使用原始特征计算导数会引入噪声。平滑算法通过加权历史信息，使特征变化更加连续。

### 2. 提高 Taylor 近似精度
Taylor 展开的精度依赖于导数的准确性。平滑后的导数更能反映特征的真实变化趋势，减少高频噪声的影响。

### 3. 改善生成质量
实验表明，在某些场景下（特别是高步数推理），启用平滑可以：
- 减少生成图像的伪影（artifacts）
- 提高图像的整体一致性
- 在保持加速比的同时，缩小与原始推理的质量差距

### 4. 灵活的平滑策略
- **指数平滑（Exponential Smoothing）**：适合捕捉长期趋势，对历史信息赋予指数衰减权重
- **移动平均（Moving Average）**：适合短期平滑，对固定窗口内的特征取均值

---

## 核心原理

### 1. 缓存历史管理

TaylorSeer 维护两个时间步的缓存：
- `cache[-1]`：当前时间步的缓存（最新）
- `cache[-2]`：上一个时间步的缓存（历史）

在每次 **full compute** 步骤开始前，执行历史迁移：
```python
cache[-2] = cache[-1]  # 将当前缓存保存为历史
cache[-1] = {}         # 初始化新的当前缓存
```

### 2. 特征历史收集

在计算导数时，收集三个时间步的特征：
```python
raw_features = [F_{-2}, F_{-1}, F_0]
```
其中：
- `F_{-2}` = `cache[-2][stream][layer][module][0]`（两步前的特征）
- `F_{-1}` = `cache[-1][stream][layer][module][0]`（一步前的特征）
- `F_0` = 当前计算得到的特征

### 3. 平滑处理

#### 指数平滑（Exponential Smoothing）
```python
smoothed[0] = raw[0]
for i in range(1, len(raw)):
    smoothed[i] = alpha * raw[i] + (1 - alpha) * smoothed[i-1]
```
- `alpha`：平滑系数（0-1），越大越接近原始值，越小越平滑
- 默认值：`0.8`（Flux 早期版本为 0.7，已统一为 0.8）

#### 移动平均（Moving Average）
```python
for i in range(len(raw)):
    if i < window_size - 1:
        smoothed[i] = mean(raw[0:i+1])
    else:
        smoothed[i] = mean(raw[i-window_size+1:i+1])
```
- `window_size`：窗口大小，默认为 2

### 4. 导数计算

#### 全局平滑模式
```python
# 使用平滑后的特征计算所有导数
smoothed = smooth_function(raw_features, alpha)
updated[0] = smoothed[-1]  # 零阶：平滑后的当前特征
updated[1] = (smoothed[-1] - smoothed[-2]) / h  # 一阶导数
updated[2] = (updated[1] - cache[-1][...][1]) / h  # 二阶导数（如果 max_order >= 2）
```

#### 混合平滑模式
```python
# 一阶导数使用原始特征
updated[0] = raw[-1]
updated[1] = (raw[-1] - raw[-2]) / h

# 二阶及以上导数使用平滑特征
if len(raw) >= 3 and max_order >= 2:
    smoothed = smooth_function(raw, alpha)
    d1_now = (smoothed[-1] - smoothed[-2]) / h
    d1_prev = (smoothed[-2] - smoothed[-3]) / h
    updated[2] = (d1_now - d1_prev) / h
```

### 5. Taylor 展开（推理阶段）

在 **Taylor 步骤**，使用缓存的导数进行展开：
```python
x = current_step - last_activated_step
output = sum(cache[i] * (x^i) / i! for i in range(max_order+1))
```

---

## 实现细节

### 核心函数

#### 1. `shift_cache_history(cache_dic, current)`
**作用**：将 `cache[-1]` 迁移到 `cache[-2]`，为新的 full compute 做准备。

**调用时机**：在 `current['type'] == 'full'` 且 `use_smoothing=True` 时，在 `module_cache_init` 之前调用。

**实现**：
```python
def shift_cache_history(cache_dic, current):
    if not cache_dic.get("taylor_cache", False):
        return

    cache = cache_dic["cache"]
    s, l, m = current["stream"], current["layer"], current["module"]

    if current["step"] == 0:
        cache[-2][s][l][m] = {}  # 第一步时初始化为空
        return

    cache[-2][s][l][m] = cache[-1][s][l][m]  # 历史迁移
```

#### 2. `exponential_smoothing(features, alpha)`
**作用**：对特征列表进行指数平滑。

**实现**：
```python
def exponential_smoothing(features: list, alpha: float) -> list:
    if len(features) <= 1:
        return features
    smoothed = [features[0]]
    for i in range(1, len(features)):
        smoothed.append(alpha * features[i] + (1 - alpha) * smoothed[i - 1])
    return smoothed
```

#### 3. `moving_average_smoothing(features, window_size)`
**作用**：对特征列表进行移动平均平滑。

**实现**：
```python
def moving_average_smoothing(features: list, window_size: int = 2) -> list:
    if len(features) < window_size:
        return features
    smoothed = []
    for i in range(len(features)):
        if i < window_size - 1:
            smoothed.append(sum(features[: i + 1]) / (i + 1))
        else:
            smoothed.append(sum(features[i - window_size + 1 : i + 1]) / window_size)
    return smoothed
```

#### 4. `derivative_approximation_with_smoothing(cache_dic, current, feature, ...)`
**作用**：使用平滑后的特征计算导数（全局平滑模式）。

**关键步骤**：
1. 收集历史特征：`raw = [F_{-2}, F_{-1}, F_0]`
2. 形状检查：确保所有特征形状一致，否则回退到非平滑模式
3. 平滑处理：`smoothed = smooth_function(raw, alpha)`
4. 计算导数：
   - `updated[0] = smoothed[-1]`
   - `updated[1] = (smoothed[-1] - smoothed[-2]) / h`
   - 更高阶导数递归计算
5. 更新缓存：`cache[-1][s][l][m] = updated`

#### 5. `derivative_approximation_hybrid_smoothing(cache_dic, current, feature, ...)`
**作用**：混合平滑模式，一阶导数用原始特征，二阶及以上用平滑特征。

**关键差异**：
```python
# 一阶导数：原始特征
updated[0] = feature
updated[1] = (raw[-1] - raw[-2]) / h

# 二阶导数：平滑特征
if len(raw) >= 3 and max_order >= 2:
    smoothed = smooth_function(raw, alpha)
    d1_now = (smoothed[-1] - smoothed[-2]) / h
    d1_prev = (smoothed[-2] - smoothed[-3]) / h
    updated[2] = (d1_now - d1_prev) / h
```

#### 6. `update_cache_or_approximate(cache_dic, current, feature)`
**作用**：统一的缓存更新/Taylor 近似入口函数。

**实现**：
```python
def update_cache_or_approximate(cache_dic, current, feature):
    if current['type'] == 'full':
        use_smoothing = cache_dic.get('use_smoothing', False)
        use_hybrid = cache_dic.get('use_hybrid_smoothing', False)
        method = cache_dic.get('smoothing_method', 'exponential')
        alpha = cache_dic.get('smoothing_alpha', 0.8)

        # 顺序很重要：先迁移历史，再初始化新槽位
        if use_smoothing:
            shift_cache_history(cache_dic, current)
        module_cache_init(cache_dic, current)

        # 根据配置选择导数计算方式
        if use_smoothing and use_hybrid:
            derivative_approximation_hybrid_smoothing(cache_dic, current, feature, method, alpha)
        elif use_smoothing:
            derivative_approximation_with_smoothing(cache_dic, current, feature, method, alpha)
        else:
            derivative_approximation(cache_dic, current, feature)

        return feature  # full 模式返回原始特征
    else:
        return taylor_formula(cache_dic, current)  # Taylor 模式返回近似值
```

---

## 关键代码修改

### 1. 缓存初始化（`cache_init.py`）

**修改点**：
- 添加 `cache[-2]` 和 `cache_index[-2]` 用于历史存储
- 添加 `attn_map[-2]` 用于注意力图历史
- 添加平滑相关配置项

**代码示例**：
```python
def cache_init(self):
    cache = {}
    cache_index = {}

    # 关键修改：添加 -2 索引用于历史
    cache[-1] = {}
    cache[-2] = {}  # 新增
    cache_index[-1] = {}
    cache_index[-2] = {}  # 新增

    # 初始化所有层的缓存（包括历史）
    for stream in ['double_stream', 'single_stream']:  # 或 ['cond', 'uncond'] for QwenImage
        cache[-1][stream] = {}
        cache[-2][stream] = {}  # 新增

        for layer_idx in range(num_layers):
            for history_idx in [-1, -2]:  # 新增循环
                cache[history_idx][stream][layer_idx] = {}
                cache_index[history_idx][layer_idx] = {}

    # 平滑配置
    cache_dic['use_smoothing'] = USE_SMOOTHING
    cache_dic['use_hybrid_smoothing'] = USE_HYBRID_SMOOTHING
    cache_dic['smoothing_method'] = SMOOTHING_METHOD
    cache_dic['smoothing_alpha'] = SMOOTHING_ALPHA
    cache_dic['smoothed_derivatives'] = {}

    return cache_dic, current
```

### 2. Transformer Block Forward（`double_transformer_forward.py`）

**修改点**：
- 在 `current['type'] == 'full'` 分支中，对每个模块调用 `update_cache_or_approximate`
- 在 double block 中，需要处理 4 个模块：`img_attn`, `img_mlp`, `txt_attn`, `txt_mlp`
- 在 single block 中，只有 1 个模块：`total`

**代码示例（Double Block）**：
```python
def taylorseer_flux_double_block_forward(self, hidden_states, encoder_hidden_states, temb, ...):
    cache_dic = joint_attention_kwargs['cache_dic']
    current = joint_attention_kwargs['current']

    if current['type'] == 'full':
        # 计算注意力
        attn_output, context_attn_output = self.attn(...)

        # img_attn 模块
        current['module'] = 'img_attn'
        update_cache_or_approximate(cache_dic, current, attn_output)
        hidden_states = hidden_states + gate_msa * attn_output

        # img_mlp 模块
        current['module'] = 'img_mlp'
        ff_output = self.ff(...)
        update_cache_or_approximate(cache_dic, current, ff_output)
        hidden_states = hidden_states + gate_mlp * ff_output

        # txt_attn 模块
        current['module'] = 'txt_attn'
        update_cache_or_approximate(cache_dic, current, context_attn_output)
        encoder_hidden_states = encoder_hidden_states + c_gate_msa * context_attn_output

        # txt_mlp 模块
        current['module'] = 'txt_mlp'
        context_ff_output = self.ff_context(...)
        update_cache_or_approximate(cache_dic, current, context_ff_output)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output

    elif current['type'] == 'Taylor':
        # Taylor 近似模式
        current['module'] = 'img_attn'
        attn_output = update_cache_or_approximate(cache_dic, current, None)
        hidden_states = hidden_states + gate_msa * attn_output

        # ... 其他模块类似

    return encoder_hidden_states, hidden_states
```

**代码示例（Single Block）**：
```python
def taylorseer_flux_single_block_forward(self, hidden_states, temb, ...):
    cache_dic = joint_attention_kwargs['cache_dic']
    current = joint_attention_kwargs['current']

    current['module'] = 'total'  # single block 只有一个模块

    if current['type'] == 'full':
        # 计算完整的 attn + mlp
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        attn_output = self.attn(...)
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states = self.proj_out(hidden_states)

        # 更新缓存
        update_cache_or_approximate(cache_dic, current, hidden_states)
    else:
        # Taylor 近似
        hidden_states = update_cache_or_approximate(cache_dic, current, None)

    hidden_states = gate * hidden_states
    hidden_states = residual + hidden_states

    return hidden_states
```

### 3. 环境变量配置

**在脚本或代码开头添加**：
```python
import os

USE_SMOOTHING = os.environ.get("USE_SMOOTHING", "False").lower() in ("true", "1", "yes")
USE_HYBRID_SMOOTHING = os.environ.get("USE_HYBRID_SMOOTHING", "False").lower() == "true"
SMOOTHING_METHOD = os.environ.get("SMOOTHING_METHOD", "exponential")
SMOOTHING_ALPHA = float(os.environ.get("SMOOTHING_ALPHA", "0.8"))
```

**在 shell 脚本中设置**：
```bash
export USE_SMOOTHING="True"
export SMOOTHING_METHOD="exponential"
export SMOOTHING_ALPHA="0.8"
# export USE_HYBRID_SMOOTHING="True"  # 可选
```

---

## 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `USE_SMOOTHING` | bool | `False` | 是否启用平滑功能 |
| `USE_HYBRID_SMOOTHING` | bool | `False` | 是否使用混合平滑（仅当 `USE_SMOOTHING=True` 时有效） |
| `SMOOTHING_METHOD` | str | `"exponential"` | 平滑方法：`"exponential"` 或 `"moving_average"` |
| `SMOOTHING_ALPHA` | float | `0.8` | 指数平滑系数（0-1），越大越接近原始值 |
| `TS_CACHE_INTERVAL` | int | `6` | 缓存间隔（N），每 N 步执行一次 full compute |
| `TS_MAX_ORDER` | int | `1` | Taylor 展开最大阶数（O） |
| `TS_FIRST_ENHANCE` | int | `3` | 前 N 步强制 full compute |

**推荐配置**：
- **标准平滑**：`USE_SMOOTHING=True`, `SMOOTHING_METHOD=exponential`, `SMOOTHING_ALPHA=0.8`
- **混合平滑**：`USE_SMOOTHING=True`, `USE_HYBRID_SMOOTHING=True`, `SMOOTHING_ALPHA=0.8`
- **移动平均**：`USE_SMOOTHING=True`, `SMOOTHING_METHOD=moving_average`

---

## 迁移指南

### 迁移到新架构（如 Flux 原生库）

#### 步骤 1：准备核心函数

将以下函数复制到你的项目中（可以放在 `taylorseer_utils.py` 或类似文件）：

1. `shift_cache_history(cache_dic, current)`
2. `exponential_smoothing(features, alpha)`
3. `moving_average_smoothing(features, window_size)`
4. `derivative_approximation(cache_dic, current, feature)` （基础版本）
5. `derivative_approximation_with_smoothing(cache_dic, current, feature, method, alpha)`
6. `derivative_approximation_hybrid_smoothing(cache_dic, current, feature, method, alpha)`
7. `taylor_formula(cache_dic, current)`
8. `module_cache_init(cache_dic, current)`
9. `update_cache_or_approximate(cache_dic, current, feature)` （统一入口）

**参考实现**：`taylorseer_core/math.py` 和 `taylorseer_core/forward_utils.py`

#### 步骤 2：修改缓存初始化

在你的 `cache_init` 函数中：

1. 添加 `cache[-2]` 和 `cache_index[-2]`
2. 为所有 stream 和 layer 初始化 `-2` 索引
3. 添加平滑配置项到 `cache_dic`

```python
# 关键修改
cache[-2] = {}
cache_index[-2] = {}

for stream in streams:
    cache[-2][stream] = {}
    for layer in range(num_layers):
        cache[-2][stream][layer] = {}

cache_dic['use_smoothing'] = USE_SMOOTHING
cache_dic['use_hybrid_smoothing'] = USE_HYBRID_SMOOTHING
cache_dic['smoothing_method'] = SMOOTHING_METHOD
cache_dic['smoothing_alpha'] = SMOOTHING_ALPHA
```

#### 步骤 3：修改 Transformer Block Forward

在每个需要缓存的模块位置：

**原始代码**：
```python
if current['type'] == 'full':
    output = self.some_layer(input)
    # 直接使用 output
    hidden_states = hidden_states + output
elif current['type'] == 'Taylor':
    output = taylor_formula(cache_dic, current)
    hidden_states = hidden_states + output
```

**修改后**：
```python
current['module'] = 'module_name'  # 设置模块名

if current['type'] == 'full':
    output = self.some_layer(input)
    update_cache_or_approximate(cache_dic, current, output)  # 统一入口
    hidden_states = hidden_states + output
elif current['type'] == 'Taylor':
    output = update_cache_or_approximate(cache_dic, current, None)  # 统一入口
    hidden_states = hidden_states + output
```

#### 步骤 4：配置环境变量

在运行脚本中添加：
```bash
export USE_SMOOTHING="True"
export SMOOTHING_METHOD="exponential"
export SMOOTHING_ALPHA="0.8"
```

#### 步骤 5：测试验证

1. **语法测试**：`--steps 1` 快速验证代码无语法错误
2. **逻辑测试**：`--steps 50` 完整推理，检查输出质量
3. **对比测试**：与非平滑版本对比，验证平滑效果

### 常见问题

#### Q1: 形状不匹配错误（Shape Mismatch）
**原因**：不同时间步的特征形状不一致（如动态 batch size 或序列长度）

**解决方案**：
- 启用形状检查：`export TS_DEBUG_SHAPES=1`
- 代码会自动回退到非平滑模式
- 或者在 `derivative_approximation_with_smoothing` 中添加形状检查逻辑

#### Q2: 平滑效果不明显
**原因**：`alpha` 值过大（接近 1），平滑程度不够

**解决方案**：
- 降低 `alpha` 值，如 `0.6` 或 `0.7`
- 或尝试移动平均：`SMOOTHING_METHOD=moving_average`

#### Q3: 生成质量下降
**原因**：过度平滑导致细节丢失

**解决方案**：
- 提高 `alpha` 值，如 `0.9`
- 或使用混合平滑：`USE_HYBRID_SMOOTHING=True`

#### Q4: 内存占用增加
**原因**：`cache[-2]` 额外存储历史特征

**解决方案**：
- 这是必要的开销，无法避免
- 如果内存紧张，可以禁用平滑：`USE_SMOOTHING=False`

---

## 总结

TaylorSeer Smooth 算法通过以下机制提升缓存加速的质量：

1. **历史管理**：维护 `cache[-1]` 和 `cache[-2]` 两个时间步的缓存
2. **平滑处理**：对特征序列应用指数平滑或移动平均
3. **导数计算**：基于平滑后的特征计算更稳定的导数
4. **灵活配置**：支持全局平滑和混合平滑两种模式

**关键优势**：
- 即插即用，只需修改缓存初始化和 forward 函数
- 配置灵活，通过环境变量控制
- 性能开销小，主要是额外的历史存储

**适用场景**：
- 高步数推理（50+ steps）
- 对生成质量要求较高的任务
- 需要在加速和质量之间取得平衡的场景

**不适用场景**：
- 低步数推理（< 20 steps），平滑效果不明显
- 内存极度受限的环境
- 对推理速度要求极高，无法接受任何额外开销的场景
