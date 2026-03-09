import argparse
import os
import re
import time
from typing import List, Optional

import torch

from inference_utils import get_torch_dtype, setup_pipeline


def sanitize_filename(text: str, max_length: int = 80) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("/", "-")
    text = re.sub(r"[^\w\-\s]", "", text)
    text = text.replace(" ", "_")
    if len(text) == 0:
        text = "prompt"
    return text[:max_length]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference with TaylorSeer caching")
    parser.add_argument("--prompt_file", type=str, required=True,
                        help="Path to a .txt file with one prompt per line.")
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unique_seed_per_prompt", action="store_true")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--prefix", type=str, default="TaylorSeer")
    parser.add_argument("--enable_cpu_offload", action="store_true")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--strategy", type=str, default="taylorseer", choices=["taylorseer"])
    parser.add_argument("--model_name", type=str, default="flux",
                        help="Adapter name for patch_model_with_cache, e.g. 'flux' or 'qwen_image'.")
    return parser.parse_args()


def read_prompts(prompt_file: str, max_images: Optional[int] = None) -> List[str]:
    with open(prompt_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    prompts = [line for line in lines if len(line) > 0]
    if max_images is not None:
        prompts = prompts[:max_images]
    return prompts


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    torch_dtype = get_torch_dtype(args.dtype)
    if args.device == "cpu" and torch_dtype in (torch.float16, torch.bfloat16):
        print("Requested low-precision dtype on CPU; overriding to float32.")
        torch_dtype = torch.float32

    if args.enable_cpu_offload:
        raise NotImplementedError("CPU offload is not supported for TaylorSeer yet.")

    print(f"Loading pipeline: {args.model} (dtype={torch_dtype}, device={args.device})")
    pipeline = setup_pipeline(args.model, int(args.steps), args.strategy, args.model_name,
                              torch_dtype, args.device)

    prompts = read_prompts(args.prompt_file, args.max_images)
    if len(prompts) == 0:
        print("No prompts found. Exiting.")
        return

    total_time_s = 0.0
    is_cuda = args.device == "cuda" and torch.cuda.is_available()
    if is_cuda:
        parameter_peak_memory = torch.cuda.max_memory_allocated(device="cuda")
        torch.cuda.reset_peak_memory_stats()

    for index, prompt in enumerate(prompts):
        effective_seed = args.seed + index if args.unique_seed_per_prompt else args.seed
        generator = torch.Generator(device="cpu").manual_seed(effective_seed)

        if is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start_time = time.time()

        image = pipeline(
            prompt,
            num_inference_steps=int(args.steps),
            generator=generator,
            height=args.height,
            width=args.width,
            guidance_scale=float(args.guidance_scale),
        ).images[0]

        if is_cuda:
            end.record()
            torch.cuda.synchronize()
            elapsed_time_s = start.elapsed_time(end) * 1e-3
        else:
            elapsed_time_s = time.time() - start_time

        total_time_s += elapsed_time_s
        safe = sanitize_filename(prompt)
        save_path = os.path.join(args.outdir, f"{args.prefix}_{index:04d}_{safe}.png")
        image.save(save_path)
        print(f"Saved: {save_path} | time: {elapsed_time_s:.2f}s")

    num_images = len(prompts)
    if is_cuda:
        peak_memory = torch.cuda.max_memory_allocated(device="cuda")
        print(f"Processed {num_images} images | avg time: {total_time_s / max(num_images, 1):.2f}s | "
              f"parameter memory: {parameter_peak_memory/1e9:.2f} GB | peak memory: {peak_memory/1e9:.2f} GB")
    else:
        print(f"Processed {num_images} images | avg time: {total_time_s / max(num_images, 1):.2f}s")


if __name__ == "__main__":
    main()

