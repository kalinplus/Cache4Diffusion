# Cache4Diffusion — Issue 记录

记录格式：`closed` 区放已解决的问题（含分析 / 改动 / 验证），`open` 区放待办（困难或需人机协同的给出建议）。

---

## closed

### 1. qwen / flux 等默认 width / height 是 512 还是 1024？应为后者
**结论**：所有入口脚本默认分辨率都已是 ≥1024，没有任何脚本默认 512。顺手修掉了一个相关 latent bug。

**排查（默认值）**：
- `flux_diffusers/taylorseer_flux/diffusers_taylorseer_flux.py`：`--height/--width` 默认 **1024**。
- `flux/taylorseer/src/sample.py`、`sample_ddp.py`：默认 **1024**。
- `qwen/taylorseer/sample.py`、`sample_ddp.py`：默认 **1328**（Qwen-Image 原生分辨率）。
- `run.py` 中各 runner 的 `defaults`：flux=1024、qwen/qwen_image=1328。
- 全仓 grep 无 `default=512` 的分辨率参数。

**发现的 latent bug（已修）**：
`run.py` 的 `qwen_image` runner 会把 `--width/--height` 透传给入口脚本，但
`qwen_image/taylorseer_qwen_image/diffusers_taylorseer_qwen_image.py` 原本**没有**定义这两个参数
（pipeline 调用里的 `width=/height=` 还被注释掉了），导致 `argparse: unrecognized arguments --width/--height` 直接报错。

**改动**：
- `qwen_image/.../diffusers_taylorseer_qwen_image.py`：新增 `--width/--height`（默认 1328），并在 `pipeline(...)` 调用中真正传入。
- `QwenImagePipeline.__call__` 确认支持 `height/width`，已实测可被接受并进入生成。

**验证**：`python run.py --model qwen_image --gpu 0 --prompt "a cute cat" --steps 50 --dry_run`
现可正常透传 `--width 1328 --height 1328`；实跑时管线成功加载并进入第 0 步（随后命中**另一处**与分辨率无关的缓存 bug，见 open 区“qwen_image diffusers 缓存未接线”）。

---

### 2. `evaluate/evaluate.py` 报 `HFValidationError: Repo id must be in the form ...`
**根因**：默认的 `--clip_model_path` / `--imagereward_model_path` 写死为本机不存在的 `/mnt/data0/...` 路径。`from_pretrained` 找不到本地目录后，把绝对路径当成 HF repo id 解析，于是抛出误导性的 `HFValidationError`。

**改动（`evaluate/evaluate.py`）**：
1. 新增 `_resolve_model_source(value, label)`：本地路径存在则原样用；形如 `namespace/name` 的当作 HF repo id（走 `HF_ENDPOINT` 镜像下载）；既不是本地路径也不像 repo id（如不存在的绝对路径）→ 抛**清晰的 `FileNotFoundError`** 并给出修复建议，不再出现晦涩的 `HFValidationError`。
2. 默认路径改为本机已有的本地副本：
   - CLIP → `/mnt/workspace/hkl/models/openai/clip-vit-large-patch14`（本机没有 `laion/CLIP-ViT-g-14`，改用本地 OpenAI ViT-L/14；如需 laion 大模型可 `--clip_model_path laion/CLIP-ViT-g-14-laion2B-s12B-b42K` 走镜像下载）。
   - ImageReward → `/mnt/workspace/hkl/models/zai-org/ImageReward`（含 `ImageReward.pt` + `med_config.json`）。
3. 支持环境变量覆盖：`EVAL_CLIP_MODEL_PATH` / `EVAL_IMAGEREWARD_MODEL_PATH`。

**验证**：`conda run -n eval python evaluate/evaluate.py --test_folder outputs/qwen/taylorseer/S50_N6O2F3 --prompt_file <单行 prompt>` 成功输出
`ClipScore=24.2358, ImageReward=0.5626`（无 reference 故 PSNR/SSIM/LPIPS=0，符合预期）。CLIP / ImageReward 均从本地加载，不再报错。

> 注：`run.py` 的 `--eval` 会在生成后自动调用此脚本；上述改动同样生效。

---

### 3. `python run.py --model qwen --gpu 0 --prompt "a cat" --steps 50` 报 `FileNotFoundError: 'img.jpg'`
**根因**：`qwen/taylorseer/sample.py` 中 `--input_image` 默认值为 `'img.jpg'`，且在 `__main__` 里**无条件**执行 `image = Image.open(args.input_image)`。文生图（`qwen-image`）根本不需要输入图，于是去打开不存在的 `img.jpg` 而崩溃。

**改动（`qwen/taylorseer/sample.py` + 同目录 `sample_ddp.py`，二者同构）**：
1. `--input_image` 默认值由 `'img.jpg'` 改为 `None`。
2. 只在 `model_name == 'qwen-image-edit'`（图编辑，唯一用到 `opts.image` 的分支）时才 `Image.open`，且缺失时给出明确报错；文生图置 `image=None`。
3. `SamplingOptions.image` 类型注解改为 `Image.Image | None`。

**验证**：`python run.py --model qwen --gpu 4 --prompt "a cute cat" --steps 50 --width 768 --height 768`
成功生成并保存 `outputs/qwen/taylorseer/S50_N6O2F3/img_0.jpg`（159KB，9.68s），全程无 `img.jpg` 报错。
（1328/1024 分辨率会 OOM，属**另一处**问题 #5，见 open 区。）

> 其余 qwen 变体（`toca/duca/freqca/teacache/qwen/baseline`）的 `sample.py` 存在完全相同的 `default='img.jpg'` 模式，但未接入 `run.py` 的默认 `qwen` runner，本次未改；如需统一，可按上述同样手法批量处理。

---

## open（困难 / 需人机协同，给出建议与分析）

### 4. HunyuanVideo：当前仓库“只支持单图 / 不足”，想直接用 `/mnt/cpfs/hkl/TaylorSeer/TaylorSeer-HunyuanVideo`
**现状**：`hunyuan_video/taylorseer_hunyuan_video/` 是基于 **diffusers** 的 TaylorSeer 实现（`diffusers_taylorseer_hunyuan_video.py` 支持 `--video-length/fps`、`export_to_video`），通过 `run.py` 的 `hunyuan_video` runner 在 `hyv15` 环境运行。用户认为能力不足。
对照外部仓库 `/mnt/cpfs/hkl/TaylorSeer/TaylorSeer-HunyuanVideo`（Tencent HunyuanVideo 完整 fork + TaylorSeer）：原生 PyTorch 模型定义、**xfuser 多卡并行**（`torchrun --ulysses-degree`）、**fp8**、CPU offload、VBench 评测、Gradio UI，能力更全。

**建议（需人选定策略后再实施，故先跳过）**：
- **方案 A（推荐，最低成本）— 直接包装外部仓库**：在 `run.py` 新增一个 runner（例如 `hunyuan_video_native`），`entry` 指向外部 `sample_video.py`，`workdir` 指向 `/mnt/cpfs/hkl/TaylorSeer/TaylorSeer-HunyuanVideo`，并为其指定**独立 conda 环境**（外部仓库的依赖：`pytorch==2.4.0 + flash-atten==2.6.3 + xfuser==0.4.0`，与 `infer`/`hyv15` 不兼容）。这样能立刻产出正规视频，无需重写。把现有 diffusers 实现保留用于 caching 实验。
- **方案 B — 移植能力**：把 xfuser 多卡 / fp8 移植进现有 diffusers 实现。工作量大、风险高。
- **方案 C — 只补缺口**：先明确“不足”具体指什么（多卡？fp8？评测？分辨率？），再针对性移植最小子集。
- **注意点**：两套实现的 cache 接口/forward 覆盖方式不同，混用易出错；建议二选一，不要在 run.py 里交叉调用。

**待人决策**：选 A / B / C。若选 A，我可继续补 `run.py` 的 `hunyuan_video_native` runner 草稿（含环境声明与命令拼装）。

---

### 5. Qwen-Image 单卡放不下，能否分卡推理？（推理场景，DDP / DeepSpeed 用不了）
**现状实测**：raw 仓库 `qwen/taylorseer/sample.py` 用 `.to(device)` 把整模放单卡，**即便 1024² 也在 80GB 卡上 OOM**（本进程自身吃到 ~79GB）；768² 才能跑通。瓶颈是“权重（~20B / bf16 ≈ 40GB）+ 60 层双流 DiT 激活 + TaylorSeer 缓存张量”叠加，原生 1328² 远超单卡。

**关键认知**：`sample_ddp.py` 的 `torchrun`/DDP 是**数据并行**——每卡复制完整模型，**不能**缓解单图 OOM，只提升批量吞吐。DeepSpeed 同理（训练导向）。真正能“分卡放下一个模型”的是**模型并行 / 序列并行 / 卸载**。

**建议（按改动从小到大）**：
1. **diffusers `device_map`（首选，改动最小）**：`qwen_image/.../diffusers_taylorseer_qwen_image.py` 现在用 `device_map='cuda'`（单卡）。改为 `device_map='balanced'` 并暴露多卡（`CUDA_VISIBLE_DEVICES=0,1`），让 diffusers 把 transformer 各层水平切分到多卡。TaylorSeer 的“按层缓存”与按层切分的 device_map 天然兼容。
2. **CPU offload**：该入口已有 `--enable_cpu_offload` 桩，但当前对 TaylorSeer 直接 `raise NotImplementedError`。可接入 `pipeline.enable_model_cpu_offload()` / `enable_sequential_cpu_offload()`，单卡+内存兜底（更慢但能跑）。`hunyuan_video`/`flux_diffusers` 已有类似支持可参考。
3. **降分辨率 / tiling**：已实测 768² 可单卡跑通；若业务允许可降到 1024 配合下面的手段。
4. **量化**：`bitsandbytes` nf4/int8 权重量化，权重显存减半（仓库里 `lllyasviel/flux1-dev-bnf4` 已有 nf4 先例）。
5. **序列并行（xfuser / USP）**：与 #4 HunyuanVideo 同套思路，对 DiT 的长序列 attention 做序列并行，改动较大但显存收益最大。
6. **通用省显存**：确认 `DIFFUSERS_ATTN_BACKEND=flash`（已默认）、bf16（已默认）、可选 `torch.compile` 复用显存、激活/梯度 checkpointing。

**推荐路径**：短期走 (1) `device_map='balanced'` 双卡 + (2) 接好 `enable_model_cpu_offload`；仍紧张再加 (4) 量化或 (5) 序列并行。

---

### 6.（验证 #1 时新发现）`qwen_image` diffusers 入口的 TaylorSeer 缓存“未接线”
**现象**：`python run.py --model qwen_image ...` 能加载模型并进入第 0 个去噪步，随即崩溃：
```
forwards/qwen_image_forward.py:92  current['layer'] = index_block
TypeError: 'NoneType' object does not support item assignment
```
**根因**：`taylorseer_qwen_image_forward` 依赖一个缓存状态字典 `current`（及 `cache_dic`），但 diffusers 入口只给 transformer 挂了个空操作的 `cache_context`（返回 `nullcontext`），**从未初始化 / 传递 `cache_dic` 与 `current`**。raw 仓库是靠自定义 `pipeline_qwenimage`（接受 `cache_dic=.../current=...` kwargs）+ `pipeline_with_cache` + `cache_init` 来注入的；而 diffusers 入口用的是 stock `DiffusionPipeline`，不会把这些 kwargs 透传给被覆盖的 `transformer.forward`。入口里被注释掉的 `pipeline.__class__.__call__ = taylorseer_qwen_image_pipeline_call` 正是曾经尝试接线的痕迹（注释说明会 NameError + OOM）。

**建议**（需人确认设计，故暂不动）：
- 方案一：把 `current`/`cache_dic` 挂到 transformer 实例属性上（如 `self._ts_current`），forward 内部读取；在生成前由入口 `cache_init` 并赋值。
- 方案二：用自定义 `__call__`（`taylorseer_qwen_image_pipeline_call`）把缓存状态经 `attention_kwargs` 之类通道传进 transformer；需先解决注释里提到的 NameError（符号未 import）与 OOM。
- 方案三：直接复用 raw 仓库的 `pipeline_with_cache` 机制（与其重复实现，不如统一）。

**影响**：与分辨率（#1）无关；#1 的 width/height 修复本身有效，只是该缓存 bug 仍挡住 qwen_image diffusers 路径完整跑通。建议和 #5 的多卡方案一起规划（同样涉及管线/forward 接线）。
