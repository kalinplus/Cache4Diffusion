import os
import re
import time
from dataclasses import dataclass
from glob import iglob

import torch
from einops import rearrange
from PIL import ExifTags, Image
from transformers.pipelines import pipeline
from tqdm import tqdm

from flux.sampling import (
    get_noise,
    get_schedule,
    prepare,
    prepare_kontext,
    prepare_fill,
    unpack,
    denoise_cache,
)
from flux.util import configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5

NSFW_THRESHOLD = 0.85  # NSFW score threshold


@dataclass
class SamplingOptions:
    prompts: list[str]  # List of prompts
    input_image: str  # Path to the input image
    mask_path: str  # Path to the mask image
    width: int  # Image width
    height: int  # Image height
    num_steps: int  # Number of sampling steps
    guidance: float  # Guidance value
    seed: int  # Random seed
    num_images_per_prompt: int  # Number of images generated per prompt
    batch_size: int  # Batch size (batching of prompts)
    model_name: str  # Model name
    output_dir: str  # Output directory
    add_sampling_metadata: bool  # Whether to add metadata
    use_nsfw_filter: bool  # Whether to enable NSFW filter
    test_FLOPs: bool  # Whether in FLOPs test mode
    monitor_gpu_usage: bool  # Whether to monitor GPU memory usage
    interval: int  # Interval for TaylorSeer
    max_order: int  # Max order for TaylorSeer
    first_enhance: int  # First enhance steps
    benchmark: bool  # Whether to run the latency + FLOPs benchmark
    benchmark_warmup: int  # Number of untimed warmup runs
    benchmark_runs: int  # Number of timed latency runs
    benchmark_report: str  # Where to write the benchmark report


def print_smoothing_config():
    print_smoothing_config = os.environ.get("PRINT_SMOOTHING_CONFIG", "True").lower() in ("true", "1", "yes")
    if not print_smoothing_config:
        return

    use_smoothing = os.environ.get("USE_SMOOTHING", "False").lower() in ("true", "1", "yes")
    if use_smoothing:
        use_hybrid = os.environ.get("USE_HYBRID_SMOOTHING", "False").lower() in ("true", "1", "yes")
        method = os.environ.get("SMOOTHING_METHOD", "exponential")
        alpha = os.environ.get("SMOOTHING_ALPHA", "0.8")
        mode = "Hybrid Smoothing" if use_hybrid else "Global Smoothing"
        print(f"TaylorSeer Smoothing: ENABLED ({mode})")
        print(f"  Method: {method}")
        print(f"  Alpha: {alpha}")
    else:
        print("TaylorSeer Smoothing: DISABLED")


def main(opts: SamplingOptions):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_smoothing_config()

    # Optional NSFW classifier
    if opts.use_nsfw_filter:
        nsfw_classifier = pipeline(
            "image-classification", model="Falconsai/nsfw_image_detection", device=device
        )
    else:
        nsfw_classifier = None

    # Load model
    model_name = opts.model_name
    if model_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown model name: {model_name}, available options: {available}")

    if model_name == "flux-schnell":
        assert opts.num_steps == 4, "flux-schnell only supports 4 steps"

    # Ensure width and height are multiples of 16
    opts.width = 16 * (opts.width // 16)
    opts.height = 16 * (opts.height // 16)

    # Set output directory and index
    output_name = os.path.join(opts.output_dir, f"img_{{idx}}.jpg")
    if not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir)
    idx = 0  # Image index

    # Load T5 and CLIP models to GPU
    t5 = load_t5(device, max_length=256 if model_name == "flux-schnell" else 512)
    clip = load_clip(device)

    # Load model to GPU
    model = load_flow_model(model_name, device=device)
    ae = load_ae(model_name, device=device)

    # Set random seed (needed by both the benchmark path and the normal loop).
    if opts.seed is not None:
        base_seed = opts.seed
    else:
        base_seed = torch.randint(0, 2**32, (1,)).item()

    # ── Optional speed benchmark: latency + model FLOPs ────────────────────
    if getattr(opts, "benchmark", False):
        from cache4diffusion_bench import run_benchmark

        bench_prompt = opts.prompts[0]  # `prompts` local is only assigned below (line ~171); use opts.prompts here
        seed = int(base_seed)

        def _prepare_once():
            x = get_noise(1, opts.height, opts.width, device=device, dtype=torch.bfloat16, seed=seed)
            if model_name in ("flux-dev", "flux-schnell"):
                inp = prepare(t5, clip, x, [bench_prompt])
            elif model_name == "flux-dev-kontext":
                inp, _, _ = prepare_kontext(t5, clip, [bench_prompt], ae, seed=seed, device=device, img_cond_path=opts.input_image)
                inp.pop("img_cond_orig")
            else:  # flux-dev-fill
                inp = prepare_fill(t5, clip, x, [bench_prompt], ae, img_cond_path=opts.input_image, mask_path=opts.mask_path)
            timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(model_name != "flux-schnell"))
            return inp, timesteps

        inp, timesteps = _prepare_once()
        bench_kwargs = {
            "num_steps": opts.num_steps,
            "test_FLOPs": False,  # the harness wraps `model` with calflops itself
            "monitor_gpu_usage": False,
            "interval": opts.interval,
            "max_order": opts.max_order,
            "first_enhance": opts.first_enhance,
        }

        def gen_once():
            with torch.no_grad():
                x = denoise_cache(model, **inp, timesteps=timesteps, guidance=opts.guidance, **bench_kwargs)
                x = unpack(x.float(), opts.height, opts.width)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    x = ae.decode(x)
            return x

        report_path = opts.benchmark_report or os.path.join(opts.output_dir, "benchmark.txt")
        run_benchmark(
            gen_fn=gen_once,
            transformer=model,
            report_path=report_path,
            meta={
                "model": "flux", "model_name": model_name, "task": "image_gen",
                "dtype": "bfloat16", "num_steps": opts.num_steps, "seed": seed,
                "guidance": opts.guidance, "width": opts.width, "height": opts.height,
                "interval": opts.interval, "max_order": opts.max_order, "first_enhance": opts.first_enhance,
                "latency_scope": "denoise + VAE decode; text encode (T5/CLIP) done once and excluded",
            },
            warmup=opts.benchmark_warmup, runs=opts.benchmark_runs,
        )
        return

    prompts = opts.prompts

    total_images = len(prompts) * opts.num_images_per_prompt

    progress_bar = tqdm(total=total_images, desc="Generating images")

    # Compute number of prompt batches
    num_prompt_batches = (len(prompts) + opts.batch_size - 1) // opts.batch_size

    for batch_idx in range(num_prompt_batches):
        prompt_start = batch_idx * opts.batch_size
        prompt_end = min(prompt_start + opts.batch_size, len(prompts))
        batch_prompts = prompts[prompt_start:prompt_end]
        num_prompts_in_batch = len(batch_prompts)

        # Generate corresponding number of images for each prompt
        for image_idx in range(opts.num_images_per_prompt):
            # Prepare random seed
            seed = int(base_seed + idx)  # Assign a different seed for each image
            idx += num_prompts_in_batch  # Update image index

            # Prepare input
            batch_size = num_prompts_in_batch
            x = get_noise(
                batch_size,
                opts.height,
                opts.width,
                device=device,
                dtype=torch.bfloat16,
                seed=seed,
            )

            # Prepare prompts
            # batch_prompts is a list containing the prompts in the current batch
            if model_name == "flux-dev" or model_name == "flux-schnell":
                inp = prepare(t5, clip, x, batch_prompts)
            elif model_name == "flux-dev-kontext":
                inp, _, _ = prepare_kontext(
                    t5, clip, batch_prompts, ae, seed=seed, device=device, img_cond_path=opts.input_image
                )
                inp.pop("img_cond_orig")
            elif model_name == "flux-dev-fill":
                inp = prepare_fill(
                    t5, clip, x, batch_prompts, ae, img_cond_path=opts.input_image, mask_path=opts.mask_path
                )

            timesteps = get_schedule(
                opts.num_steps, inp["img"].shape[1], shift=(model_name != "flux-schnell")
            )  # type: ignore

            kwargs = {
                "num_steps": opts.num_steps,
                "test_FLOPs": opts.test_FLOPs,
                "monitor_gpu_usage": opts.monitor_gpu_usage,
                "interval": opts.interval,
                "max_order": opts.max_order,
                "first_enhance": opts.first_enhance,
            }

            # Denoising
            with torch.no_grad():
                x = denoise_cache(model, **inp, timesteps=timesteps, guidance=opts.guidance, **kwargs)

                # Decode latent variables
                x = unpack(x.float(), opts.height, opts.width)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    x = ae.decode(x)

            # Convert to PIL format and save
            x = x.clamp(-1, 1)
            x = embed_watermark(x.float())
            x = rearrange(x, "b c h w -> b h w c")

            for i in range(batch_size):
                img_array = x[i]
                img = Image.fromarray((127.5 * (img_array + 1.0)).cpu().byte().numpy())

                # Optional NSFW filtering
                if opts.use_nsfw_filter and nsfw_classifier is not None:
                    nsfw_result = nsfw_classifier(img)
                    nsfw_score = next((res["score"] for res in nsfw_result if res["label"] == "nsfw"), 0.0)  # type: ignore
                else:
                    nsfw_score = 0.0  # If the filter is not enabled, assume safe

                if nsfw_score < NSFW_THRESHOLD:
                    exif_data = Image.Exif()
                    exif_data[ExifTags.Base.Software] = (
                        "AI generated;t2i;flux"
                        if model_name == "flux-schnell" or model_name == "flux-dev"
                        else "AI generated;ti2i;flux"
                    )
                    exif_data[ExifTags.Base.Make] = "Black Forest Labs"
                    exif_data[ExifTags.Base.Model] = model_name
                    if opts.add_sampling_metadata:
                        exif_data[ExifTags.Base.ImageDescription] = batch_prompts[i]
                    # Save image
                    fn = output_name.format(idx=idx - num_prompts_in_batch + i)
                    img.save(fn, exif=exif_data, quality=95, subsampling=0)
                else:
                    print(f"Generated image may contain inappropriate content, skipped.")

                progress_bar.update(1)

    progress_bar.close()


def read_prompts(prompt_file: str):
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate images using the flux model.")
    parser.add_argument(
        "--prompt_file", type=str, default="prompts/DrawBench200.txt", help="Path to the prompt text file."
    )
    parser.add_argument("--input_image", type=str, default="img.jpg", help="Path to the input image.")
    parser.add_argument("--mask_path", type=str, default="mask.jpg", help="Path to the mask image.")  # TODO
    parser.add_argument("--width", type=int, default=1024, help="Width of the generated image.")
    parser.add_argument("--height", type=int, default=1024, help="Height of the generated image.")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of sampling steps.")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance value.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="Number of images per prompt.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (prompt batching).")
    parser.add_argument(
        "--model_name",
        type=str,
        default="flux-schnell",
        choices=["flux-schnell", "flux-dev", "flux-dev-kontext", "flux-dev-fill"],
        help="Model name.",
    )
    parser.add_argument("--output_dir", type=str, default="samples/test", help="Directory to save images.")
    parser.add_argument(
        "--add_sampling_metadata", action="store_true", help="Whether to add prompt metadata to images."
    )
    parser.add_argument("--use_nsfw_filter", action="store_true", help="Enable NSFW filter.")
    parser.add_argument("--test_FLOPs", action="store_true", help="Test inference computation cost.")
    parser.add_argument(
        "--monitor_gpu_usage", action="store_true", help="Monitor GPU memory usage during sampling."
    )

    parser.add_argument("--interval", type=int, default=4)
    parser.add_argument("--max_order", type=int, default=0)
    parser.add_argument("--first_enhance", type=int, default=1)
    # max_order 0 first_enhance 1 for FORA

    # Speed benchmark (latency + FLOPs) via the shared cache4diffusion_bench harness.
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark one generation: measure latency + model FLOPs.")
    parser.add_argument("--benchmark_warmup", type=int, default=1)
    parser.add_argument("--benchmark_runs", type=int, default=1)
    parser.add_argument("--benchmark_report", type=str, default=None)

    args = parser.parse_args()

    prompts = read_prompts(args.prompt_file)

    opts = SamplingOptions(
        prompts=prompts,
        input_image=args.input_image,
        mask_path=args.mask_path,
        width=args.width,
        height=args.height,
        num_steps=args.num_steps,
        guidance=args.guidance,
        seed=args.seed,
        num_images_per_prompt=args.num_images_per_prompt,
        batch_size=args.batch_size,
        model_name=args.model_name,
        output_dir=args.output_dir,
        add_sampling_metadata=args.add_sampling_metadata,
        use_nsfw_filter=args.use_nsfw_filter,
        test_FLOPs=args.test_FLOPs,
        monitor_gpu_usage=args.monitor_gpu_usage,
        interval=args.interval,
        max_order=args.max_order,
        first_enhance=args.first_enhance,
        benchmark=args.benchmark,
        benchmark_warmup=args.benchmark_warmup,
        benchmark_runs=args.benchmark_runs,
        benchmark_report=args.benchmark_report,
    )

    main(opts)
    # CUDA_VISIBLE_DEVICES=0 python src/sample.py
