import argparse
import os
import re
import time

import torch
from diffusers import DiffusionPipeline, FluxTransformer2DModel, BitsAndBytesConfig as DiffusersBnBConfig

from cache_functions import cache_init, cal_type
from forwards import (
    taylorseer_flux_forward,
    taylorseer_flux_single_block_forward,
    taylorseer_flux_double_block_forward,
)


def sanitize_filename(text: str, max_length: int = 80) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("/", "-")
    text = re.sub(r"[^\w\-\s]", "", text)
    text = text.replace(" ", "_")
    if len(text) == 0:
        text = "prompt"
    return text[:max_length]


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float8":
        return torch.float8_e4m3fn
    return torch.float32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-prompt inference with TaylorSeer caching")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32", "float8"])
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--prefix", type=str, default="TaylorSeer")
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--lora", type=str, default=None,
                        help="Path to a LoRA safetensors file.")
    parser.add_argument("--lora_scale", type=float, default=1.0,
                        help="LoRA weight scale (default: 1.0).")
    parser.add_argument("--transformer_file", type=str, default=None,
                        help="Path to a single-file transformer (e.g., NF4 safetensors). "
                             "If provided, the transformer will be loaded via from_single_file "
                             "and the base --model will be used for remaining pipeline components.")
    parser.add_argument("--quantize", type=str, default="none",
                        choices=["none", "nf4"],
                        help="Quantization mode for the transformer. "
                             "'nf4' loads the transformer with BitsAndBytes 4-bit NF4 quantization online.")
    parser.add_argument("--enable_cpu_offload", action="store_true")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    # Speed benchmark (latency + FLOPs) via the shared cache4diffusion_bench harness.
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark one generation: measure latency + transformer FLOPs.")
    parser.add_argument("--benchmark_warmup", type=int, default=1)
    parser.add_argument("--benchmark_runs", type=int, default=1)
    parser.add_argument("--benchmark_report", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.transformer_file and args.quantize != "none":
        raise ValueError("--transformer_file and --quantize cannot be used together.")

    torch_dtype = get_torch_dtype(args.dtype)
    if args.device == "cpu" and torch_dtype in (torch.float16, torch.bfloat16):
        print("Requested low-precision dtype on CPU; overriding to float32.")
        torch_dtype = torch.float32

    if args.enable_cpu_offload:
        raise NotImplementedError("CPU offload is not supported for TaylorSeer yet.")

    print(f"Loading pipeline: {args.model} (dtype={torch_dtype}, device={args.device})")
    if args.transformer_file:
        print(f"Loading transformer from single file: {args.transformer_file}")
        transformer = FluxTransformer2DModel.from_single_file(
            args.transformer_file,
            config=args.model,
            subfolder="transformer",
            torch_dtype=torch_dtype,
        )
        pipeline = DiffusionPipeline.from_pretrained(
            args.model,
            transformer=transformer,
            torch_dtype=torch_dtype,
        )
    elif args.quantize == "nf4":
        print("Loading transformer with online NF4 quantization ...")
        nf4_config = DiffusersBnBConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            args.model,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch_dtype,
        )
        pipeline = DiffusionPipeline.from_pretrained(
            args.model,
            transformer=transformer,
            torch_dtype=torch_dtype,
        )
    else:
        pipeline = DiffusionPipeline.from_pretrained(args.model, torch_dtype=torch_dtype)

    if args.lora:
        pipeline.load_lora_weights(args.lora)
        if args.lora_scale != 1.0:
            pipeline.set_adapters(["default"], weights=[args.lora_scale])

    # Patch transformer forward with TaylorSeer
    pipeline.transformer.__class__.num_steps = int(args.steps)
    pipeline.transformer.forward = taylorseer_flux_forward.__get__(
        pipeline.transformer, pipeline.transformer.__class__
    )
    for block in pipeline.transformer.transformer_blocks:
        block.forward = taylorseer_flux_double_block_forward.__get__(
            block, block.__class__
        )
    for block in pipeline.transformer.single_transformer_blocks:
        block.forward = taylorseer_flux_single_block_forward.__get__(
            block, block.__class__
        )

    if hasattr(pipeline, 'vae'):
        pipeline.vae.enable_tiling()

    pipeline.to(args.device)

    # ── Optional speed benchmark: latency + transformer FLOPs ──────────────
    if args.benchmark:
        from cache4diffusion_bench import run_benchmark

        def gen_once():
            return pipeline(
                args.prompt,
                num_inference_steps=int(args.steps),
                generator=torch.Generator("cpu").manual_seed(int(args.seed)),
                guidance_scale=float(args.guidance_scale),
                height=args.height,
                width=args.width,
            ).images[0]

        report_path = args.benchmark_report or os.path.join(args.outdir, "benchmark.txt")
        run_benchmark(
            gen_fn=gen_once,
            transformer=pipeline.transformer,
            report_path=report_path,
            meta={
                "model": "flux_diffusers",
                "task": "image_gen",
                "dtype": args.dtype,
                "steps": args.steps,
                "seed": args.seed,
                "guidance_scale": args.guidance_scale,
                "width": args.width,
                "height": args.height,
                "cache_interval": os.environ.get("FRESH_THRESHOLD", "1"),
                "cache_max_order": os.environ.get("MAX_ORDER", "0"),
                "cache_first_enhance": os.environ.get("FIRST_ENHANCE", "3"),
                "use_smoothing": os.environ.get("USE_SMOOTHING", "False"),
                "smoothing_alpha": os.environ.get("SMOOTHING_ALPHA", "0.7"),
            },
            warmup=args.benchmark_warmup,
            runs=args.benchmark_runs,
            save_fn=lambda img: img.save(
                os.path.join(args.outdir, f"{args.prefix}_bench.png")),
        )
        return

    is_cuda = args.device == "cuda" and torch.cuda.is_available()
    if is_cuda:
        parameter_peak_memory = torch.cuda.max_memory_allocated(device="cuda")
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start_time = time.time()

    image = pipeline(
        args.prompt,
        num_inference_steps=int(args.steps),
        generator=torch.Generator("cpu").manual_seed(int(args.seed)),
        guidance_scale=float(args.guidance_scale),
        height=args.height,
        width=args.width,
    ).images[0]

    if is_cuda:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end) * 1e-3
        peak_memory = torch.cuda.max_memory_allocated(device="cuda")
        print(f"epoch time: {elapsed_time:.2f} sec, "
              f"parameter memory: {parameter_peak_memory/1e9:.2f} GB, "
              f"memory: {peak_memory/1e9:.2f} GB")
    else:
        elapsed_time = time.time() - start_time
        print(f"elapsed time: {elapsed_time:.2f} sec")

    safe = sanitize_filename(args.prompt)
    save_path = os.path.join(args.outdir, f"{args.prefix}_{safe}.png")
    image.save(save_path)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
