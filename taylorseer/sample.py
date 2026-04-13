import os
import argparse
from dataclasses import dataclass

import torch
import torch.distributed as dist
from PIL import ExifTags, Image
from tqdm import tqdm

from cache_functions import cache_init, pipe_with_cache
from pipelines.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers import AutoencoderKL

@dataclass
class SamplingOptions:
    prompts: list[str]          # List of prompts
    height: int                 # Image height
    width: int                  # Image width
    num_steps: int              # Number of sampling steps
    guidance: float             # Guidance value
    seed: int                   # Random seed
    model_name: str             # Model name
    output_dir: str             # Output directory
    test_FLOPs: bool            # Whether in FLOPs test mode
    monitor_gpu_usage: bool     # Whether to monitor GPU memory usage
    interval: int               # Cache period length
    max_order: int              # Maximum order of Taylor expansion
    first_enhance: int          # Initial enhancement steps

def main(opts: SamplingOptions):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)

    total_prompts = len(opts.prompts)
    per_proc = (total_prompts + world_size - 1) // world_size
    start = rank * per_proc
    end = min(start + per_proc, total_prompts)
    prompts = opts.prompts[start:end]

    if rank == 0 and not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir, exist_ok=True)

    # load base model
    model_path = os.environ.get("SDXL_MODEL_PATH", "stabilityai/stable-diffusion-xl-base-1.0")
    vae = AutoencoderKL.from_pretrained(
        os.environ.get("SDXL_VAE_PATH", "madebyollin/sdxl-vae-fp16-fix"),
        torch_dtype=torch.float16,
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_path,
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")
    pipe = pipe_with_cache(pipe)

    progress_bar = tqdm(total=len(prompts), desc="Generating images") if rank == 0 else None

    for i in range(len(prompts)):
        generator = torch.Generator(device).manual_seed(int(opts.seed + i))

        kwargs = {
            'height': opts.height,
            'width': opts.width,
            'num_steps': opts.num_steps,
            'test_FLOPs': opts.test_FLOPs,
            'monitor_gpu_usage': opts.monitor_gpu_usage,
            'interval': opts.interval,
            'max_order': opts.max_order,
            'first_enhance': opts.first_enhance,
        }
        cache_dic, current = cache_init(**kwargs)

        image = pipe(
            prompt=prompts[i],
            height=opts.height,
            width=opts.width,
            num_inference_steps=opts.num_steps,
            guidance_scale=opts.guidance,
            generator=generator,
            cache_dic=cache_dic,
            current=current,
        ).images[0] # type: ignore

        exif_data = Image.Exif()
        exif_data[ExifTags.Base.ImageDescription] = prompts[i]
        image_path = os.path.join(opts.output_dir, f"img_{start + i}.jpg")
        image.save(image_path, exif=exif_data, quality=95, subsampling=0)

        if progress_bar is not None:
            progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()

    dist.barrier()
    dist.destroy_process_group()


def read_prompts(prompt_file: str):
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate images using the stable-diffusion-xl model.")
    parser.add_argument('--prompt_file', type=str, default='prompts/DrawBench200.txt', help='Path to the prompt text file.')
    parser.add_argument('--height', type=int, default=1024, help='Height of the generated image.')
    parser.add_argument('--width', type=int, default=1024, help='Width of the generated image.')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of sampling steps.')
    parser.add_argument('--guidance', type=float, default=5.0, help='Guidance value.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--model_name', type=str, default='stable-diffusion-xl', choices=['stable-diffusion-xl'], help='Model name.')
    parser.add_argument('--output_dir', type=str, default='samples/test', help='Directory to save images.')
    parser.add_argument('--test_FLOPs', action='store_true', help='Test inference computation cost.')
    parser.add_argument('--monitor_gpu_usage', action='store_true', help='Monitor GPU memory usage during sampling.')

    parser.add_argument('--interval', type=int, default=6)
    parser.add_argument('--max_order', type=int, default=2)
    parser.add_argument('--first_enhance', type=int, default=3)

    args = parser.parse_args()
    prompts = read_prompts(args.prompt_file)

    opts = SamplingOptions(
        prompts=prompts,
        height=args.height,
        width=args.width,
        num_steps=args.num_steps,
        guidance=args.guidance,
        seed=args.seed,
        model_name=args.model_name,
        output_dir=args.output_dir,
        test_FLOPs=args.test_FLOPs,
        monitor_gpu_usage=args.monitor_gpu_usage,
        interval=args.interval,
        max_order=args.max_order,
        first_enhance=args.first_enhance,
    )

    main(opts)
    # CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 sample.py
