# Qwen TaylorSeer Benchmark Notes

本文档记录 `qwen/taylorseer` 当前测速脚本的实现方式，供后续其它模型或方法复用。

## 目标

测速时需要分别得到：

- latency：单张图片端到端生成耗时。
- FLOPs/MACs：TaylorSeer denoising loop 中 transformer forward 的理论计算量。
- GPU memory：latency 阶段的峰值显存。

当前实现采用单独的 benchmark 入口：

```text
qwen/taylorseer/benchmark_sample.py
```

普通出图入口 `sample.py` 不参与 benchmark 改造，避免影响原有生成流程。

## 当前默认配置

当前 `qwen/taylorseer/run_sample.sh` 中配置为：

```bash
--num_warmup_prompts 1
--num_benchmark_prompts 1
--num_flops_prompts 1
--test_FLOPs
--monitor_gpu_usage
```

含义：

- warmup 跑 1 张，不记录指标。
- latency 跑 1 张，保存图片并记录耗时。
- FLOPs 跑 1 张，只统计 FLOPs，不保存图片。

如果要获得更稳定的 latency 均值，可以把 `--num_benchmark_prompts` 改成 5 或 10。FLOPs 通常不需要跑很多张，因为固定分辨率和步数时主要差异来自 prompt token 长度。

## 为什么 latency 和 FLOPs 分开测

`calflops.calculate_flops(...)` 的便利接口不是读取一次已经发生的 forward，而是：

```text
注册 hooks -> 额外执行一次 model.forward -> 统计 FLOPs -> 移除 hooks
```

如果在 latency 阶段直接使用 `calculate_flops(...)`，会把额外 forward 的开销算进 latency，导致耗时偏大；显存也会明显增加。

因此当前实现分成两段：

```text
latency run: test_flops=False，真实生成并保存图片
FLOPs run: test_flops=True，单独统计 FLOPs，不保存图片
```

## Latency 测量流程

代码位置：`benchmark_sample.py::main()` 的 `Latency benchmark` loop。

流程：

```python
sync_cuda()
start = time.perf_counter()

result, cache_dic = call_pipeline(
    ...,
    test_flops=False,
    monitor_gpu_usage=args.monitor_gpu_usage,
    output_type="pil",
)

sync_cuda()
latency_sec = time.perf_counter() - start
```

计时范围：

- 包含一次 `pipe(...)` 调用。
- 不包含 `from_pretrained(...)` 模型加载。
- 不包含 `img.save(...)` 图片保存。
- 不包含写 `benchmark.txt`。

CUDA 同步是必要的，否则 `time.perf_counter()` 只会测到 kernel launch 时间，不是实际 GPU 执行完成时间。

latency 阶段保存图片：

```python
saved_files = save_images(result, args, prompt=prompt, image_index=i)
```

保存发生在计时结束之后，所以磁盘 IO 不污染 latency。

## FLOPs 测量流程

代码位置：

```text
qwen/taylorseer/pipeline/pipeline_qwenimage.py
qwen/taylorseer/pipeline/pipeline_qwenimage_edit.py
```

`benchmark_sample.py` 在 FLOPs 阶段调用：

```python
result, cache_dic = call_pipeline(
    ...,
    test_flops=True,
    monitor_gpu_usage=False,
    output_type="latent",
)
```

`test_flops=True` 会写入：

```python
cache_dic["test_FLOPs"] = True
```

pipeline 里据此启用 FLOPs 统计。

当前实现没有使用 `calculate_flops(...)` 便利接口，而是用低层 hook：

```python
from calflops.calculate_pipline import CalFlopsPipline

flops_counter = CalFlopsPipline(
    model=self.transformer,
    include_backPropagation=False,
    compute_bp_factor=2.0,
    is_sparse=False,
)
flops_counter.start_flops_calculate()

try:
    noise_pred = self.transformer(...)[0]
    flops = flops_counter.get_total_flops()
    macs = flops_counter.get_total_macs()
    params = flops_counter.get_total_params()
finally:
    flops_counter.end_flops_calculate()
```

这样 FLOPs hook 挂在真实的 transformer forward 上，不会额外执行第二次 forward。

统计范围：

- 统计 transformer denoising loop。
- 不统计模型加载。
- 不统计 text encoder。
- 不统计 VAE decode。
- 不统计图片保存。

这是有意为之：TaylorSeer 主要影响 denoising transformer 的计算路径，比较 cache 方法时应优先看这一部分。

## 备选：直接使用 calculate_flops

也可以直接使用 `calflops.calculate_flops(...)` 便利函数。这个方式适合写一个纯 FLOPs profiling 入口：只准备 transformer 输入并统计 FLOPs，不保存图片，最好也不做 VAE decode。

需要注意：`calculate_flops(...)` 本身会执行一次 `model.forward()`，所以它不是零开销统计。不要把它放进 latency 计时段。

基本调用形式：

```python
from calflops import calculate_flops

with torch.no_grad():
    flops, macs, params = calculate_flops(
        model=self.transformer,
        kwargs={
            "hidden_states": latents,
            "timestep": timestep / 1000,
            "guidance": guidance,
            "encoder_hidden_states_mask": prompt_embeds_mask,
            "encoder_hidden_states": prompt_embeds,
            "img_shapes": img_shapes,
            "txt_seq_lens": txt_seq_lens,
            "attention_kwargs": self.attention_kwargs,
            "return_dict": False,
            "cache_dic": cache_dic,
            "current": current,
        },
        print_results=False,
        print_detailed=False,
    )
```

如果要统计完整 denoising loop 的 transformer FLOPs，需要在 pipeline 已经准备好 prompt embeddings、latents、timesteps 之后循环每个 step：

```python
total_flops = 0
total_macs = 0
total_params = 0

current["step"] = 0

for t in timesteps:
    timestep = t.expand(latents.shape[0]).to(latents.dtype)

    cal_type(cache_dic=cache_dic, current=current)

    with self.transformer.cache_context("cond"):
        current["stream"] = "cond"
        flops, macs, params = calculate_flops(
            model=self.transformer,
            kwargs={
                "hidden_states": latents,
                "timestep": timestep / 1000,
                "guidance": guidance,
                "encoder_hidden_states_mask": prompt_embeds_mask,
                "encoder_hidden_states": prompt_embeds,
                "img_shapes": img_shapes,
                "txt_seq_lens": txt_seq_lens,
                "attention_kwargs": self.attention_kwargs,
                "return_dict": False,
                "cache_dic": cache_dic,
                "current": current,
            },
            print_results=False,
            print_detailed=False,
        )
        total_flops += float(flops)
        total_macs += float(macs)
        total_params += float(params)

    current["step"] += 1
```

如果启用了 true CFG，还需要对 `uncond` stream 再调用一次：

```python
with self.transformer.cache_context("uncond"):
    current["stream"] = "uncond"
    flops, macs, params = calculate_flops(
        model=self.transformer,
        kwargs={
            "hidden_states": latents,
            "timestep": timestep / 1000,
            "guidance": guidance,
            "encoder_hidden_states_mask": negative_prompt_embeds_mask,
            "encoder_hidden_states": negative_prompt_embeds,
            "img_shapes": img_shapes,
            "txt_seq_lens": negative_txt_seq_lens,
            "attention_kwargs": self.attention_kwargs,
            "return_dict": False,
            "cache_dic": cache_dic,
            "current": current,
        },
        print_results=False,
        print_detailed=False,
    )
```

这个流程可以“不生图”，但不能完全跳过 pipeline 前半部分，因为 transformer 输入需要由 pipeline 准备：

- prompt / negative prompt embeddings。
- prompt masks。
- initial latents。
- timesteps。
- guidance tensor。
- img shapes / txt sequence lengths。
- TaylorSeer `cache_dic` 和 `current`。

推荐做法是新增一个 `flops_only` pipeline 分支：

```text
load pipeline
prepare prompt embeddings / latents / timesteps
loop timesteps
  cal_type(...)
  calculate_flops(self.transformer, kwargs=...)
write total_flops / total_macs / params
return，不 decode VAE，不 save image
```

这样可以避免生成 PIL 图片和磁盘 IO，但每个 step 仍会执行一次 transformer forward，因为这是 `calculate_flops(...)` 的工作方式。

当前代码最终没有采用这个便利函数方案，而是用 `CalFlopsPipline` hook 包住真实 forward，原因是：

- `calculate_flops(...)` 会额外 forward；如果同时还要真实生成，会增加显存和时间。
- hook 真实 forward 可以避免重复执行 transformer。
- 当前 benchmark 需要同时保留一个可出图的 latency 路径和一个 FLOPs 路径，hook 方案对显存更友好。

## sample.py --test_FLOPs 与 benchmark_sample.py 的区别

`sample.py` 里的 `--test_FLOPs` 只是把开关传入 `cache_dic`：

```python
cache_dic["test_FLOPs"] = opts.test_FLOPs
```

真正的 FLOPs 统计仍发生在 pipeline 里。当前 pipeline 已改成 `CalFlopsPipline` hook 真实 transformer forward，所以 `sample.py --test_FLOPs` 不会再额外执行第二次 transformer forward。

但两者用途不同：

- `sample.py --test_FLOPs`：正常生图，同时打印/记录 FLOPs。适合临时看单次生成计算量。
- `benchmark_sample.py`：专门测速，分开执行 warmup、latency、FLOPs，并把结果写入 `benchmark.txt`。

显存处理也不同：

- `sample.py` 不会在每个 prompt 后显式 `del result, cache_dic`、`gc.collect()`、`torch.cuda.empty_cache()`。
- `benchmark_sample.py` 每轮后会释放本轮引用并清理 allocator cache，连续测多个 prompt 时更稳。

因此如果要做正式 latency/FLOPs 表格，优先用 `benchmark_sample.py`。如果只想正常出图并顺便看 FLOPs，可以用 `sample.py --test_FLOPs`，但不要把这时的 wall time 当作纯 latency。

## 显存处理

每轮 warmup、latency、FLOPs 后都会释放本轮引用：

```python
del result, cache_dic
release_cuda_memory()
```

`release_cuda_memory()` 会执行：

```python
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
```

这样可以避免上一轮的 TaylorSeer cache 张量继续被 Python 局部变量持有，降低下一轮 OOM 风险。

注意：`empty_cache()` 只能释放 PyTorch allocator 中未被引用的缓存块；如果还有 Python 引用持有张量，它不会释放那部分显存。

## 输出文件

每组配置输出到独立目录，例如：

```text
samples/qwen/taylorseer/benchmark_db200/N6O1F3Alpha0/
```

目录中包含：

```text
img_0.jpg
benchmark.txt
```

`benchmark.txt` 记录：

- model path。
- prompt file。
- interval / max_order / first_enhance。
- smoothing 环境变量。
- latency scope 和 FLOPs scope。
- per-image latency。
- average latency。
- per-prompt FLOPs/MACs/Params。
- average FLOPs/MACs。
- peak GPU memory。

## Smoothing 与 TaylorSeer 参数

TaylorSeer 参数通过命令行传入：

```bash
--interval "${interval}"
--max_order "${max_order}"
--first_enhance "${first_enhance}"
```

smoothing 通过环境变量控制：

```bash
export USE_SMOOTHING="False"       # alpha=0
export USE_SMOOTHING="True"        # alpha!=0
export SMOOTHING_ALPHA="$alpha"
export SMOOTHING_METHOD="exponential"
```

`run_sample.sh` 使用循环生成输出目录：

```bash
output_dir=".../N${interval}O${max_order}F${first_enhance}Alpha${alpha}"
```

这样不同配置的图片和 `benchmark.txt` 不会互相覆盖。

## 本地模型路径

Qwen-Image 本地路径通过脚本显式传入：

```bash
export QWEN_IMAGE_MODEL_PATH="/mnt/data1/pretrained_models/Qwen/Qwen-Image"

python benchmark_sample.py \
    --model_path "${QWEN_IMAGE_MODEL_PATH}" \
    ...
```

`benchmark_sample.py` 也支持环境变量默认值：

```text
QWEN_IMAGE_MODEL_PATH
QWEN_IMAGE_EDIT_MODEL_PATH
```

## 复用建议

迁移到其它测速脚本时，遵循以下原则：

1. latency 和 FLOPs 分开跑。
2. latency 阶段关闭 FLOPs hook。
3. latency 只包住 `pipe(...)`，不要包模型加载和图片保存。
4. GPU latency 前后都要 `torch.cuda.synchronize()`。
5. FLOPs 优先用 hook 包真实 forward，避免 `calculate_flops(...)` 额外 forward。
6. 每轮后释放 result/cache 引用，再 `gc.collect()` 和 `torch.cuda.empty_cache()`。
7. 把所有参数和环境变量写入 report，保证后续结果可追溯。
