## 任务卡 v2（阶段2：流程工程化）— TeaCache-Lite for HunyuanImage-2.1

### 背景

TaylorSeer-Lite 已在 `HunyuanImage-2.1/scripts/taylorseer_lite_hyimage/` 实现并验证，采用"缓存 final block 输出 + Taylor 外推"的 skip 策略。现在需要实现 TeaCache-Lite，作为实验对比组。TeaCache-Lite 共享同一缓存点（final block 输出），但 skip 步策略不同：用 L1 阈值动态判定 + 残差累加，而非固定 interval + Taylor 外推。

仓库中已有 TeaCache 的 FLUX 参考实现（`flux/teacache/src/flux/model.py`），可直接参考其判定逻辑和 cache 结构。HunyuanImage 的 DiT 结构与 FLUX 相似（double_blocks → single_blocks → final_layer），但额外支持 MeanFlow、byT5 glyph、CFG 等特性，需保持兼容。

### 最终目标

TeaCache-Lite 可通过 `--use_teacache_lite` 参数启用，与 TaylorSeer-Lite 共用同一个 batch 入口脚本，生成 200 张图片用于评测。

### 分步计划（有序，每步独立可验收）

Step 1: 创建 `teacache_lite_hyimage` 模块骨架 + cache_utils
  - 产出物:
    - `scripts/teacache_lite_hyimage/__init__.py`
    - `scripts/teacache_lite_hyimage/cache_utils/__init__.py`
    - `scripts/teacache_lite_hyimage/cache_utils/cache_init.py`
  - 内容:
    - `cache_init(num_steps, rel_l1_thresh=0.6, coefficients=None)` 返回 `(cache_dic, current)`
    - `cache_dic` 包含: `num_steps`, `rel_l1_thresh`, `coefficients` (4阶多项式), `enable_teacache=True`
    - `current` 包含: `cnt=0`, `accumulated_rel_l1_distance=0.0`, `previous_modulated_input=None`, `previous_residual=None`
    - `rel_l1_thresh` 和 `coefficients` 支持环境变量 `TEACACHE_REL_L1_THRESH` / `TEACACHE_COEFFICIENTS` 覆盖
    - 默认 coefficients: `[4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01]` (from flux TeaCache)
  - 验收: `python -c "from scripts.teacache_lite_hyimage.cache_utils import cache_init; print(cache_init(50))"` 无报错

Step 2: 实现 DiT forward patch（核心逻辑）
  - 产出物: `scripts/teacache_lite_hyimage/forwards/apply_teacache_lite_hyimage_forward.py`
  - 内容:
    - `apply_teacache_lite_hyimage_forward(model)` 函数，用 `MethodType` patch `model.forward`
    - forward 签名与 TaylorSeer 版完全一致（`hidden_states, timestep, text_states, encoder_attention_mask, ..., cache_dic, current`）
    - 前置处理与 TaylorSeer 版完全一致（img_in, time_in, MeanFlow, guidance, text_projection, byT5, cu_seqlens）
    - **TeaCache 判定逻辑**（位于 block 循环之前）:
      1. 计算 `modulated_input`: 用 `self.double_blocks[0].img_norm1(img.clone())` 和 `self.double_blocks[0].img_mod(vec.clone())` 得到 scale/shift
      2. `rel_l1 = |modulated_t - modulated_{t-1}| / |modulated_{t-1}|`（用 `.abs().mean()`）
      3. `accumulated += np.poly1d(coefficients)(rel_l1)` （4阶多项式缩放）
      4. 第一步 (`cnt==0`) 和最后一步 (`cnt==num_steps-1`) 强制 `should_calc=True`，`accumulated=0`
      5. 其余步: `should_calc = (accumulated >= rel_l1_thresh)`；若 True 则 `accumulated=0`
    - **分支执行**:
      - `should_calc=False`: `img += current['previous_residual']`，跳过所有 block
      - `should_calc=True`: 保存 `ori_img = img.clone()`，走完全部 double_blocks → single_blocks → final_layer，计算 `current['previous_residual'] = img - ori_img`
    - 后置处理（unpatchify）与 TaylorSeer 版一致
    - **重要**: final_layer 在两个分支中都要调用（skip 分支的 img 加了 residual 后也需要 unpatchify 之前的 final_layer？——参考 flux 实现：skip 分支不加 final_layer，residual 是 final_layer 之前的；需要验证）
  - 验收: 语法检查通过，import 无报错

Step 3: 实现 pipeline __call__ patch
  - 产出物: `scripts/teacache_lite_hyimage/forwards/apply_teacache_lite_hyimage_pipeline.py`
  - 内容:
    - `apply_teacache_lite_hyimage_pipeline(pipeline)` 函数
    - 定义 `_denoise_step_teacache_lite`，传递 `cache_dic` 和 `current` 给 DiT forward（与 TaylorSeer 版结构一致）
    - 定义 `__call_teacache_lite_pipeline`，替换 `pipeline.__class__.__call__`
    - 初始化用 `from scripts.teacache_lite_hyimage.cache_utils import cache_init`
    - `cache_dic, current = cache_init(sampling_steps, rel_l1_thresh=rel_l1_thresh)`
    - **无需** `use_smoothing` 相关参数（TeaCache 没有平滑机制）
    - 其余逻辑（CFG, MeanFlow, timestep, latents step）与 TaylorSeer 版完全一致
    - patch 方式: `pipeline._denoise_step_teacache_lite = MethodType(...)` + `pipeline.__class__.__call__ = __call_teacache_lite_pipeline`
  - 验收: import 无报错

Step 4: 修改 batch 入口脚本支持 TeaCache-Lite
  - 产出物: 修改 `HunyuanImage-2.1/run_hyimage_taylorseer_lite_batch.py`
  - 内容:
    - 添加 `--use_teacache_lite` 参数 (action='store_true', default=False)
    - 添加 `--rel_l1_thresh` 参数 (type=float, default=0.6)
    - 在模型加载后，根据 flag 选择 patch 方式:
      ```python
      if args.use_teacache_lite:
          from scripts.teacache_lite_hyimage.forwards.apply_teacache_lite_hyimage_pipeline import apply_teacache_lite_hyimage_pipeline
          from scripts.teacache_lite_hyimage.forwards.apply_teacache_lite_hyimage_forward import apply_teacache_lite_hyimage_forward
          apply_teacache_lite_hyimage_pipeline(pipe, rel_l1_thresh=args.rel_l1_thresh)
          apply_teacache_lite_hyimage_forward(pipe.dit)
      elif args.use_taylorseer_lite:
          # existing logic
      ```
    - pipeline 调用时，TeaCache 不传 `use_smoothing` 参数
  - 验收: `python -c "import argparse; ..."` 参数解析正确

Step 5: 编写单张推理 shell 脚本
  - 产出物: `HunyuanImage-2.1/scripts/infer_teacache_single_hunyuanimage.sh`
  - 内容: 参照 `infer_taylorseer_single_hunyuanimage.sh`，改用 `--use_teacache_lite` + `--rel_l1_thresh 0.6`
  - 验收: shell 脚本语法正确

Step 6: 编写 batch 推理 shell 脚本
  - 产出物: `HunyuanImage-2.1/scripts/infer_teacache_batch_hunyuanimage.sh`
  - 内容: 参照 `infer_taylorseer_batch_hunyuanimage.sh`，改用 `--use_teacache_lite --rel_l1_thresh 0.6 --prefix TeaCache_lambda0.6`
  - 验收: shell 脚本语法正确


### 非目标
- 不修改 TaylorSeer-Lite 的任何已有代码
- 不修改 `evaluate.py`（文件名 pattern 追加在生成完成后单独处理）
- 不修改 `run_hyimage_taylorseer_lite.py`（单张入口，TeaCache 只走 batch）
- 不引入新 Python 依赖（TeaCache 仅用 numpy 的 `np.poly1d`，已在环境内）
- 不实现 TeaCache 的 FLOPs 统计和 GPU 监控功能
- 不实现 smoothing（TeaCache 不需要）

### 参考
- 风格参考: `scripts/taylorseer_lite_hyimage/` 全部文件（目录结构、patch 模式、命名风格）
- TeaCache 判定逻辑: `flux/teacache/src/flux/model.py` L114-L150
- TeaCache cache_init: `flux/teacache/src/flux/modules/cache_functions/cache_init.py`
- Pipeline patch 参考: `scripts/taylorseer_lite_hyimage/forwards/apply_taylorseer_lite_hyimage_pipeline.py`
- Forward patch 参考: `scripts/taylorseer_lite_hyimage/forwards/apply_taylorseer_lite_hyimage_forward.py`
- Batch 入口: `HunyuanImage-2.1/run_hyimage_taylorseer_lite_batch.py`
- Shell 脚本: `scripts/infer_taylorseer_single_hunyuanimage.sh`, `scripts/infer_taylorseer_batch_hunyuanimage.sh`

### 自动化验收命令

运行环境: conda activate qwenimage (HunyuanImage 模型所在环境)
执行命令格式: conda run -n qwenimage python ...

每步完成后可直接运行以下命令验收：

- Step1: `conma run -n qwenimage python -c "from scripts.teacache_lite_hyimage.cache_utils import cache_init; cache_dic, current = cache_init(50); print(cache_dic); print(current)"`
- Step2: `conda run -n qwenimage python -c "from scripts.teacache_lite_hyimage.forwards.apply_teacache_lite_hyimage_forward import apply_teacache_lite_hyimage_forward; print('OK')"`
- Step3: `conda run -n qwenimage python -c "from scripts.teacache_lite_hyimage.forwards.apply_teacache_lite_hyimage_pipeline import apply_teacache_lite_hyimage_pipeline; print('OK')"`
- Step4: `conda run -n qwenimage python HunyuanImage-2.1/run_hyimage_taylorseer_lite_batch.py --help`
- Step5-6: `bash -n HunyuanImage-2.1/scripts/infer_teacache_single_hunyuanimage.sh && bash -n HunyuanImage-2.1/scripts/infer_teacache_batch_hunyuanimage.sh`
- End-to-end: `bash HunyuanImage-2.1/scripts/infer_teacache_single_hunyuanimage.sh`（生成单张图片，确认输出不是纯黑/噪声）

### 成功条件
- 所有步骤验收命令通过（exit code 0）
- diff 范围仅在 `scripts/teacache_lite_hyimage/` 和 `run_hyimage_taylorseer_lite_batch.py` 和新 shell 脚本
- 单张推理生成图片有大致形状（非纯黑/纯噪声）
- 与 TaylorSeer-Lite 的 prefix/命名不冲突

### 错误处理约定
- 如某步失败：先分析原因，给出修复方案，等确认后再修
- 如连续两次失败：停下来，列出可能原因，不要继续盲目重试
- 如遇到环境/依赖问题：报告具体报错，不要自行修改环境配置

---

在开始实施之前，请先：
1. 用你自己的话复述：目标是什么、边界是什么
2. 列出你认为的风险点或歧义
3. 给出最小改动方案（只写思路，不写代码）
4. 等我确认后再实施
