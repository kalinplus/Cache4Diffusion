import argparse
import gc
import math
import os
import statistics
import time

import torch
from PIL import ExifTags, Image
from tqdm import tqdm
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from cache_functions import cache_init
from cache_functions.cache_utils import pipeline_with_cache
from pipeline.pipeline_qwenimage import QwenImagePipeline
from pipeline.pipeline_qwenimage_edit import QwenImageEditPipeline


def read_prompts(prompt_file: str):
    with open(prompt_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def release_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_pipeline(args, device: str):
    if args.model_name == "qwen-image":
        pipe = QwenImagePipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
        ).to(device=device)
    elif args.model_name == "qwen-image-edit":
        pipe = QwenImageEditPipeline.from_pretrained(
            args.edit_model_path,
            torch_dtype=torch.bfloat16,
        ).to(device=device)
    elif args.model_name == "qwen-image-lightning":
        assert args.num_steps == 8, "qwen-image-lightning only supports 8 steps."
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        pipe = QwenImagePipeline.from_pretrained(
            args.model_path,
            scheduler=scheduler,
            torch_dtype=torch.bfloat16,
        ).to(device=device)
        pipe.load_lora_weights(
            "lightx2v/Qwen-Image-Lightning",
            weight_name="Qwen-Image-Lightning-8steps-V2.0.safetensors",
        )
    else:
        raise ValueError(f"Model name {args.model_name} not supported.")

    return pipeline_with_cache(pipe)


def build_cache(args, test_flops: bool, monitor_gpu_usage: bool):
    return cache_init(kwargs={
        "num_steps": args.num_steps,
        "test_FLOPs": test_flops,
        "monitor_gpu_usage": monitor_gpu_usage,
        "interval": args.interval,
        "max_order": args.max_order,
        "first_enhance": args.first_enhance,
    })


def call_pipeline(pipe, args, image, prompt: str, seed: int, device: str, test_flops: bool, monitor_gpu_usage: bool, output_type: str):
    cache_dic, current = build_cache(args, test_flops=test_flops, monitor_gpu_usage=monitor_gpu_usage)
    generators = [torch.Generator(device).manual_seed(int(seed))]

    common_kwargs = dict(
        prompt=[prompt],
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        generator=generators,
        output_type=output_type,
        cache_dic=cache_dic,
        current=current,
    )

    if args.model_name == "qwen-image":
        result = pipe(
            height=args.height,
            width=args.width,
            **common_kwargs,
        )
    elif args.model_name == "qwen-image-edit":
        result = pipe(
            image=image,
            height=args.height,
            width=args.width,
            **common_kwargs,
        )
    elif args.model_name == "qwen-image-lightning":
        result = pipe(
            height=1024,
            width=1024,
            true_cfg_scale=1.0,
            **common_kwargs,
        )
    else:
        raise ValueError(f"Model name {args.model_name} not supported.")

    return result, cache_dic


def extract_images(result):
    images = getattr(result, "images", None)
    if images is None:
        if isinstance(result, (list, tuple)):
            images = list(result)
        else:
            images = [result]
    return [img for img in images if isinstance(img, Image.Image)]


def save_images(result, args, prompt: str, image_index: int):
    saved_files = []
    for offset, img in enumerate(extract_images(result)):
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Software] = (
            "AI generated;t2i;qwen" if args.model_name != "qwen-image-edit" else "AI generated;ti2i;qwen"
        )
        exif_data[ExifTags.Base.Make] = "Qwen"
        exif_data[ExifTags.Base.Model] = args.model_name
        if args.add_sampling_metadata:
            exif_data[ExifTags.Base.ImageDescription] = prompt

        filename = os.path.join(args.output_dir, f"img_{image_index + offset}.jpg")
        img.save(filename, exif=exif_data, quality=95, subsampling=0)
        saved_files.append(filename)
    return saved_files


def mean_or_none(values):
    return statistics.mean(values) if values else None


def fmt(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}"


def write_report(args, latency_records, flops_records, report_path):
    latency_values = [item["latency_sec"] for item in latency_records]
    flops_values = [item["flops"] for item in flops_records if item.get("flops") is not None]
    macs_values = [item["macs"] for item in flops_records if item.get("macs") is not None]
    params_values = [item["params"] for item in flops_records if item.get("params") is not None]
    peak_values = [
        item["gpu_memory_peak_gb"]
        for item in latency_records
        if item.get("gpu_memory_peak_gb") is not None
    ]

    lines = [
        "Qwen TaylorSeer benchmark",
        "",
        f"model_name: {args.model_name}",
        f"model_path: {args.model_path}",
        f"edit_model_path: {args.edit_model_path}",
        f"prompt_file: {args.prompt_file}",
        f"output_dir: {args.output_dir}",
        f"num_warmup_prompts: {args.actual_num_warmup_prompts}",
        f"num_benchmark_prompts: {len(latency_records)}",
        f"num_flops_prompts: {len(flops_records)}",
        f"width: {args.width}",
        f"height: {args.height}",
        f"num_steps: {args.num_steps}",
        f"guidance_scale: {args.guidance_scale}",
        f"seed: {args.seed}",
        f"interval: {args.interval}",
        f"max_order: {args.max_order}",
        f"first_enhance: {args.first_enhance}",
        f"use_smoothing: {os.environ.get('USE_SMOOTHING', 'False')}",
        f"smoothing_method: {os.environ.get('SMOOTHING_METHOD', 'exponential')}",
        f"smoothing_alpha: {os.environ.get('SMOOTHING_ALPHA', '0.8')}",
        "latency_scope: pipe call only; excludes model loading and image saving",
        "flops_scope: transformer denoising only; measured in a separate profiling run",
        "",
        "Summary",
        f"average_latency_sec_per_image: {fmt(mean_or_none(latency_values), 6)}",
        f"average_flops_T_per_image: {fmt(mean_or_none(flops_values) * 1e-12 if flops_values else None, 6)}",
        f"average_macs_T_per_image: {fmt(mean_or_none(macs_values) * 1e-12 if macs_values else None, 6)}",
        f"params_G: {fmt(mean_or_none(params_values) * 1e-9 if params_values else None, 6)}",
        f"average_peak_gpu_memory_gb: {fmt(mean_or_none(peak_values), 6)}",
        "",
        "Latency records",
    ]

    for item in latency_records:
        saved = ",".join(item["saved_files"]) if item["saved_files"] else "N/A"
        lines.append(
            f"{item['index']}\tlatency_sec={item['latency_sec']:.6f}"
            f"\tpeak_gpu_gb={fmt(item.get('gpu_memory_peak_gb'), 6)}"
            f"\tseed={item['seed']}\tfile={saved}\tprompt={item['prompt']}"
        )

    lines.extend(["", "FLOPs records"])
    for item in flops_records:
        lines.append(
            f"{item['index']}\tflops_T={fmt(item.get('flops') * 1e-12 if item.get('flops') is not None else None, 6)}"
            f"\tmacs_T={fmt(item.get('macs') * 1e-12 if item.get('macs') is not None else None, 6)}"
            f"\tparams_G={fmt(item.get('params') * 1e-9 if item.get('params') is not None else None, 6)}"
            f"\tseed={item['seed']}\tprompt={item['prompt']}"
        )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    prompts = read_prompts(args.prompt_file)
    if not prompts:
        raise ValueError(f"No prompts found in {args.prompt_file}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.open(args.input_image)
    pipe = load_pipeline(args, device=device)

    warmup_prompts = prompts[: min(args.num_warmup_prompts, len(prompts))]
    latency_prompts = prompts[: min(args.num_benchmark_prompts, len(prompts))]
    flops_prompts = prompts[: min(args.num_flops_prompts, len(prompts))] if args.test_FLOPs else []
    args.actual_num_warmup_prompts = len(warmup_prompts)

    base_seed = args.seed if args.seed is not None else torch.randint(0, 2**32, (1,)).item()

    for i, prompt in enumerate(tqdm(warmup_prompts, desc="Warmup")):
        result, cache_dic = call_pipeline(
            pipe,
            args,
            image,
            prompt=prompt,
            seed=base_seed + i,
            device=device,
            test_flops=False,
            monitor_gpu_usage=False,
            output_type="pil",
        )
        del result, cache_dic
        release_cuda_memory()

    latency_records = []
    for i, prompt in enumerate(tqdm(latency_prompts, desc="Latency benchmark")):
        seed = base_seed + i
        sync_cuda()
        start = time.perf_counter()
        result, cache_dic = call_pipeline(
            pipe,
            args,
            image,
            prompt=prompt,
            seed=seed,
            device=device,
            test_flops=False,
            monitor_gpu_usage=args.monitor_gpu_usage,
            output_type="pil",
        )
        sync_cuda()
        latency_sec = time.perf_counter() - start

        saved_files = save_images(result, args, prompt=prompt, image_index=i)
        metrics = cache_dic.get("metrics", {})
        latency_records.append({
            "index": i,
            "prompt": prompt,
            "seed": seed,
            "latency_sec": latency_sec,
            "saved_files": saved_files,
            "gpu_memory_peak_gb": metrics.get("gpu_memory_peak_gb"),
        })
        del result, cache_dic
        release_cuda_memory()

    flops_records = []
    release_cuda_memory()
    for i, prompt in enumerate(tqdm(flops_prompts, desc="FLOPs benchmark")):
        seed = base_seed + i
        result, cache_dic = call_pipeline(
            pipe,
            args,
            image,
            prompt=prompt,
            seed=seed,
            device=device,
            test_flops=True,
            monitor_gpu_usage=False,
            output_type="latent",
        )
        metrics = cache_dic.get("metrics", {})
        flops_records.append({
            "index": i,
            "prompt": prompt,
            "seed": seed,
            "flops": metrics.get("flops"),
            "macs": metrics.get("macs"),
            "params": metrics.get("params"),
        })
        del result, cache_dic
        release_cuda_memory()

    report_path = os.path.join(args.output_dir, args.benchmark_report)
    write_report(args, latency_records, flops_records, report_path)
    print(f"Saved {len(latency_records)} images in {args.output_dir}")
    print(f"Benchmark report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Qwen TaylorSeer latency and FLOPs.")
    parser.add_argument("--input_image", type=str, default="img.jpg", help="Path to the input image.")
    parser.add_argument("--prompt_file", type=str, default="prompts/DrawBench200.txt", help="Path to the prompt text file.")
    parser.add_argument("--negative_prompt", type=str, default=" ", help="Negative prompt for guidance.")
    parser.add_argument("--width", type=int, default=1328, help="Width of the generated image.")
    parser.add_argument("--height", type=int, default=1328, help="Height of the generated image.")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of sampling steps.")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--model_name", type=str, default="qwen-image", choices=["qwen-image", "qwen-image-edit", "qwen-image-lightning"], help="Model name.")
    parser.add_argument("--model_path", type=str, default=os.environ.get("QWEN_IMAGE_MODEL_PATH", "Qwen/Qwen-Image"), help="Path or Hugging Face id for Qwen-Image.")
    parser.add_argument("--edit_model_path", type=str, default=os.environ.get("QWEN_IMAGE_EDIT_MODEL_PATH", "Qwen/Qwen-Image-Edit"), help="Path or Hugging Face id for Qwen-Image-Edit.")
    parser.add_argument("--output_dir", type=str, default="samples/benchmark", help="Directory to save images and benchmark report.")
    parser.add_argument("--add_sampling_metadata", action="store_true", help="Whether to add prompt metadata to images.")
    parser.add_argument("--test_FLOPs", action="store_true", help="Run a separate FLOPs benchmark stage.")
    parser.add_argument("--monitor_gpu_usage", action="store_true", help="Monitor GPU memory usage during latency sampling.")
    parser.add_argument("--num_warmup_prompts", type=int, default=1, help="Number of warmup prompts, not measured.")
    parser.add_argument("--num_benchmark_prompts", type=int, default=1, help="Number of prompts used for latency benchmark.")
    parser.add_argument("--num_flops_prompts", type=int, default=1, help="Number of prompts used for FLOPs benchmark.")
    parser.add_argument("--benchmark_report", type=str, default="benchmark.txt", help="Benchmark report filename inside output_dir.")
    parser.add_argument("--interval", type=int, default=6)
    parser.add_argument("--max_order", type=int, default=2)
    parser.add_argument("--first_enhance", type=int, default=3)

    main(parser.parse_args())
