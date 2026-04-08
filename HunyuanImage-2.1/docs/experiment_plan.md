# HunyuanImage-2.1 Acceleration Experiment Plan

## Background

TaylorSeer-Lite 已实现、生成、评测完毕。
现需补充 TeaCache-Lite 和 step reduction baseline，形成完整对比。

TaylorSeer-Lite 机制回顾（与 Full 版的核心区别见下方 "Why Lite" 小节）：
- Full 步：走完全部 block，**只在 final block 输出处缓存一次**，计算 Taylor 系数（函数值 + 差分导数）
- Taylor 步：跳过全部 block，用 Taylor 公式外推当前步的 final output（`f(x₀) + f'(x₀)·Δx + ...`）
- 调度：前 `first_enhance` 步强制 full，之后每隔 `cache_interval` 步 full 一次

## Method Analysis

### Why Lite (cache only at final output)?

**Lite vs Full 的核心区别**在于缓存粒度：
- **TaylorSeer (Full)**：缓存每个 block 的每个 module（attn + mlp/moe）输出 → `n_order × n_blocks × n_modules` 个张量，VRAM 开销大
- **TaylorSeer-Lite**：**只在 final block 输出处缓存一次** → 彻底消除 `n_blocks × n_modules` 的倍增，VRAM 大幅节省，无需编译优化 cache I/O

三种方法的 Lite 变体共享同一缓存点（final block 输出），区别在于 skip 步的预测策略：

| Method | Skip 策略 | 说明 |
|--------|----------|------|
| TaylorSeer | Taylor 展开外推 | `output = f(x₀) + f'(x₀)·Δx + f''(x₀)·Δx²/2!`，导数由相邻 full 步差分近似 |
| FORA | 直接复用缓存 | 返回上次 full 步的 cached output，不做 Taylor 展开或任何预测 |
| TeaCache | 残差累加 | `img += previous_residual`，residual = 上次 full 步的 `output - input` |

### FORA ≡ TaylorSeer(max_order=0)

**FORA 不需要单独实现。** 当 `TS_MAX_ORDER=0` 时，TaylorSeer-Lite 退化为 FORA：

- `derivative_approximation`：`range(0)` 不执行，缓存只存 `{0: feature}`，无导数
- `taylor_formula`：`output = (1/0!) * cache[0] * (x^0) = feature`，与距离无关，等价于直接复用

因此 FORA 实验组直接复用 TaylorSeer-Lite 脚本，仅需设置环境变量：

```bash
export TS_MAX_ORDER=0
export TS_CACHE_INTERVAL=6
export TS_FIRST_ENHANCE=1
```

ToCa（attention 分数驱动 token 选择）和 DuCa（随机 token 选择）的核心是 **per-token 粒度的部分计算**，在 Lite 层无法忠实还原，故不实现。

| Method | Lite 等价性 | 原因 |
|--------|------------|------|
| FORA | `TS_MAX_ORDER=0` 即可 | TaylorSeer 的 0 阶展开 = 直接复用缓存 |
| TeaCache | 需新实现 | 残差累加 + L1 阈值动态判定，非 interval 调度 |
| ToCa | 不可行 | 依赖 per-token attention 分数 + MLP 选择性计算 |
| DuCa | 不可行 | 依赖 per-token 随机选择 + DuCa/FORA 交替调度 |

### TeaCache-Lite 核心逻辑

```
每步在 DiT forward 前：
1. 计算 modulated_input = img_norm * (1 + scale) + shift
2. rel_l1 = |modulated_t - modulated_{t-1}| / |modulated_{t-1}|
3. accumulated += polynomial_4th_order(rel_l1)   # 4 阶多项式缩放
4. if accumulated < threshold:
       img += previous_residual                   # skip 步：加 residual，不调 DiT
   else:
       正常走 DiT
       residual = output - input                   # 更新 residual
       accumulated = 0
5. 第一步和最后一步强制 compute。
```

与 TaylorSeer-Lite 的区别：skip 步用 residual 累加而非 Taylor 外推，且是否 skip 由 L1 阈值动态判定（非固定 interval）。

## Experiment Matrix

### New Implementations (1)

| Method | 复用基础 | 改动量 |
|--------|---------|--------|
| FORA-Lite | TaylorSeer-Lite（`TS_MAX_ORDER=0`） | 无代码改动，仅环境变量 |
| TeaCache-Lite | TaylorSeer-Lite pipeline 框架 | 新写 modulated_input 计算 + L1 累积 + 残差机制 |

### Experiments (6 groups)

| # | Method | Param | Target Speedup | Action |
|---|--------|-------|---------------|--------|
| 0 | Step reduction | 50 steps | 1x (baseline) | reference folder, 确认是否已生成 |
| 1 | Step reduction | 17 steps | ~3x | 改 `--steps` 参数 |
| 2 | Step reduction | 10 steps | ~5x | 改 `--steps` 参数 |
| 3 | FORA-Lite | N=6 | ~4.5-5x | `TS_MAX_ORDER=0 TS_CACHE_INTERVAL=6 TS_FIRST_ENHANCE=1` |
| 4 | TeaCache-Lite | lambda=0.6 | ~3.5-4x | 新实现 |
| 5 | TeaCache-Lite | lambda=0.8 | ~4.5-5x | 新实现 |

已有结果：TaylorSeer（多组 smoothing 参数）— 评测已收集完毕。

### Evaluation

- **Metrics**: CLIP Score, ImageReward, PSNR, SSIM, LPIPS
- **Reference**: 50-step baseline (group #0) 作为 PSNR/SSIM/LPIPS 的 reference folder
- **Prompts**: DrawBench200 (200 prompts)
- **Tool**: `HunyuanImage-2.1/evaluate.py`
- **Conda env**: `eval`

### Output Naming Convention

沿用现有 evaluate.py 的文件名匹配规则（4 位 prompt index）：

```
Origin:          HunyuanImage_0000_Aredcoloredcar.png
TaylorSeer:     TS_no_smooth_N4O1F3_0000_Aredcoloredcar.png
Step reduction: Step17_0000_Aredcoloredcar.png
FORA:           FORA_N6_0000_Aredcoloredcar.png
TeaCache:       TeaCache_lambda0.6_0000_Aredcoloredcar.png
```

需要在 evaluate.py 中添加对应的文件名解析模式。

## Implementation Plan

### 1. FORA-Lite

无需新代码。直接使用 TaylorSeer-Lite 脚本，通过环境变量控制：

```bash
export TS_MAX_ORDER=0 TS_CACHE_INTERVAL=6 TS_FIRST_ENHANCE=1
```

Shell 脚本中设置这些变量即可，prefix 设为 `FORA_N6`。

### 2. TeaCache-Lite

**Files to create:**

```
HunyuanImage-2.1/scripts/teacache_lite_hyimage/
    __init__.py
    cache_utils.py             # 初始化 cache_dic (threshold, coefficients, residual, etc.)
    forwards/
        apply_teacache_lite_hyimage_pipeline.py  # patch pipeline __call__
        apply_teacache_lite_hyimage_forward.py   # patch DiT forward (modulated input + L1)
```

**DiT forward patch** (`apply_teacache_lite_hyimage_forward.py`):
- 在 block 循环前计算 `modulated_input`
- 与 `previous_modulated_input` 比较，计算 rel_l1
- 多项式缩放后累加到 accumulated
- 判定 should_calc
- should_calc=False：跳过所有 block，`img += previous_residual`
- should_calc=True：走完全部 blocks，更新 residual 和 accumulated
- 第一步和最后一步强制 compute
- 参数：`TEACACHE_REL_L1_THRESH` (env var, 即 lambda)

**Polynomial coefficients** (from flux TeaCache, 需验证是否适用于 HunyuanImage):
```
[4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
```

### 3. Batch Inference Script

沿用现有模式，为每种方法写独立 shell 脚本（参考 `infer_taylorseer_batch_hunyuanimage.sh`）。

### 4. evaluate.py 更新

在文件名解析中添加 FORA 和 TeaCache 的 pattern：

```python
# FORA: FORA_N6_0000_Aredcoloredcar.png
# TeaCache: TeaCache_lambda0.6_0000_Aredcoloredcar.png
# Step reduction: Step17_0000_Aredcoloredcar.png
```

## Work Items

| # | Task | Status |
|---|------|--------|
| 1 | 确认 50-step baseline 是否已生成 | Pending |
| 2 | FORA-Lite：写 shell 脚本（`TS_MAX_ORDER=0`） | Pending |
| 3 | 实现 TeaCache-Lite | Pending |
| 4 | TeaCache-Lite：写 batch 推理脚本 | Pending |
| 5 | 更新 evaluate.py 文件名解析 | Pending |
| 6 | 生成 6 组 x 200 张图 | Pending |
| 7 | 运行评测，汇总结果 | Pending |
