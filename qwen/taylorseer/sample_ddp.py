import os
import math
import torch
from PIL import Image, ExifTags
from tqdm import tqdm
from dataclasses import dataclass
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler 
import torch.distributed as dist

from pipeline.pipeline_qwenimage import QwenImagePipeline
from pipeline.pipeline_qwenimage_edit import QwenImageEditPipeline
from cache_functions.cache_utils import pipeline_with_cache
from cache_functions import cache_init


@dataclass
class SamplingOptions:
    image: Image.Image          # Input image
    prompts: list[str]          # List of prompts
    negative_prompt: str        # Negative prompt for guidance
    width: int                  # Image width
    height: int                 # Image height
    num_steps: int              # Number of sampling steps
    guidance_scale: float       # Guidance scale
    seed: int | None            # Random seed
    num_images_per_prompt: int  # Number of images generated per prompt
    batch_size: int             # Batch size (batching of prompts)
    model_name: str             # Model name
    output_dir: str             # Output directory
    add_sampling_metadata: bool # Whether to add metadata
    test_FLOPs: bool            # Whether in FLOPs test mode
    monitor_gpu_usage: bool    # Whether to monitor GPU memory usage
    interval: int               # Cache period length
    max_order: int              # Maximum order of Taylor expansion
    first_enhance: int          # Initial enhancement steps


def main(opts: SamplingOptions):
     
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    total_prompts = len(opts.prompts)
    per_proc = (total_prompts + world_size - 1) // world_size
    start = rank * per_proc
    end = min(start + per_proc, total_prompts)
    local_prompts = opts.prompts[start:end]

    if rank == 0:
        if not os.path.exists(opts.output_dir):
            os.makedirs(opts.output_dir, exist_ok=True)

    # Load pipeline
    if opts.model_name == 'qwen-image':
        pipe = QwenImagePipeline.from_pretrained(
            "Qwen/Qwen-Image", 
            torch_dtype=torch.bfloat16
        ).to(device=device)
    elif opts.model_name == 'qwen-image-edit':
        pipe = QwenImageEditPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit", 
            torch_dtype=torch.bfloat16
        ).to(device=device)
    elif opts.model_name == 'qwen-image-lightning':
        assert opts.num_steps == 8, "qwen-image-lightning only supports 8 steps."
        
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),  # We use shift=3 in distillation
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),  # We use shift=3 in distillation
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,  # set shift_terminal to None
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

        pipe = QwenImagePipeline.from_pretrained(
            "Qwen/Qwen-Image", 
            scheduler=scheduler,
            torch_dtype=torch.bfloat16
        ).to(device=device)

        pipe.load_lora_weights(
            "lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-8steps-V2.0.safetensors"
        )
    else:
        raise ValueError(f"Model name {opts.model_name} not supported.")

    pipe = pipeline_with_cache(pipe)

    total_images = len(opts.prompts) * opts.num_images_per_prompt
    local_images = len(local_prompts) * opts.num_images_per_prompt

    progress_bar = tqdm(total=local_images, desc="Generating images")

    num_prompt_batches = (len(local_prompts) + opts.batch_size - 1) // opts.batch_size

    for batch_idx in range(num_prompt_batches):
        prompt_start = batch_idx * opts.batch_size
        prompt_end = min(prompt_start + opts.batch_size, len(local_prompts))
        batch_prompts = local_prompts[prompt_start:prompt_end]
        num_prompts_in_batch = len(batch_prompts)

        # Generate corresponding number of images for each prompt
        for image_idx in range(opts.num_images_per_prompt):
            generators = []
            for i in range(num_prompts_in_batch):
                global_prompt_idx = start + prompt_start + i
                global_img_idx = global_prompt_idx * opts.num_images_per_prompt + image_idx

                if opts.seed is not None:
                    seed = opts.seed + global_img_idx
                else:
                    seed = torch.randint(0, 2**32, (1,)).item()
                
                generator = torch.Generator(device).manual_seed(int(seed))
                generators.append(generator)
            
            # Initialize cache
            cache_dic, current = cache_init(kwargs={
                'num_steps': opts.num_steps,      
                'test_FLOPs': opts.test_FLOPs,
                'monitor_gpu_usage': opts.monitor_gpu_usage,
                'interval': opts.interval, 
                'max_order': opts.max_order,
                'first_enhance': opts.first_enhance,
            })
            
            # Generate images 
            if opts.model_name == 'qwen-image':
                result = pipe(
                    prompt=batch_prompts,
                    negative_prompt=opts.negative_prompt,
                    height=opts.height,
                    width=opts.width,
                    num_inference_steps=opts.num_steps,
                    guidance_scale=opts.guidance_scale,
                    generator=generators,
                    cache_dic=cache_dic,
                    current=current
                )
            elif opts.model_name == 'qwen-image-edit':
                result = pipe(
                    image=opts.image, # type: ignore
                    prompt=batch_prompts,
                    negative_prompt=opts.negative_prompt,
                    height=opts.height,
                    width=opts.width,
                    num_inference_steps=opts.num_steps,
                    guidance_scale=opts.guidance_scale,
                    generator=generators,
                    cache_dic=cache_dic,
                    current=current
                )
            elif opts.model_name == 'qwen-image-lightning':
                result = pipe(
                    prompt=batch_prompts,
                    negative_prompt=opts.negative_prompt,
                    height=1024,
                    width=1024,
                    num_inference_steps=opts.num_steps,
                    true_cfg_scale=1.0,
                    guidance_scale=opts.guidance_scale,
                    generator=generators,
                    cache_dic=cache_dic,
                    current=current
                )
            else:
                raise ValueError(f"Model name {opts.model_name} not supported.")
            
            # Handle different return types from pipeline
            images = getattr(result, 'images', None)
            if images is None:
                if isinstance(result, (list, tuple)):
                    images = list(result)
                else:
                    images = [result]
            
            for i, img in enumerate(images):
                if not isinstance(img, Image.Image):
                    continue

                # Add EXIF metadata
                exif_data = Image.Exif()
                exif_data[ExifTags.Base.Software] = "AI generated;t2i;qwen" if opts.model_name == 'qwen-image' else "AI generated;ti2i;qwen"
                exif_data[ExifTags.Base.Make] = "Qwen"
                exif_data[ExifTags.Base.Model] = opts.model_name
                if opts.add_sampling_metadata and i < len(batch_prompts):
                    exif_data[ExifTags.Base.ImageDescription] = batch_prompts[i]
                
                # Save image
                global_prompt_idx = start + prompt_start + i
                global_img_idx = global_prompt_idx * opts.num_images_per_prompt + image_idx
                filename = f"{opts.output_dir}/img_{global_img_idx}.jpg"
                img.save(filename, exif=exif_data, quality=95, subsampling=0)
                
                if rank == 0 and progress_bar is not None:
                    progress_bar.update(1)
    
    if rank == 0 and progress_bar is not None:
        progress_bar.close()

    dist.barrier()
    
    if rank == 0:
        print("All images generated.")
    
    dist.destroy_process_group()


def read_prompts(prompt_file: str):
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate images using the flux model.")
    parser.add_argument('--input_image', type=str, default='img.jpg', help='Path to the input image.')
    parser.add_argument('--prompt_file', type=str, default='prompts/DrawBench200.txt', help='Path to the prompt text file.')
    parser.add_argument('--negative_prompt', type=str, default=" ", help='Negative prompt for guidance.')
    parser.add_argument('--width', type=int, default=1328, help='Width of the generated image.')
    parser.add_argument('--height', type=int, default=1328, help='Height of the generated image.')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of sampling steps.')
    parser.add_argument('--guidance_scale', type=float, default=1.0, help='Guidance scale.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--num_images_per_prompt', type=int, default=1, help='Number of images per prompt.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (prompt batching).')
    parser.add_argument('--model_name', type=str, default='qwen-image', choices=['qwen-image', 'qwen-image-edit', 'qwen-image-lightning'], help='Model name.')
    parser.add_argument('--output_dir', type=str, default='samples/test', help='Directory to save images.')
    parser.add_argument('--add_sampling_metadata', action='store_true', help='Whether to add prompt metadata to images.')
    parser.add_argument('--test_FLOPs', action='store_true', help='Test inference computation cost.')
    parser.add_argument('--monitor_gpu_usage', action='store_true', help='Monitor GPU memory usage during sampling.')

    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--max_order', type=int, default=0)
    parser.add_argument('--first_enhance', type=int, default=1)

    args = parser.parse_args()

    image = Image.open(args.input_image)
    prompts = read_prompts(args.prompt_file)

    opts = SamplingOptions(
        image=image,
        prompts=prompts,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        num_images_per_prompt=args.num_images_per_prompt,
        batch_size=args.batch_size,
        model_name=args.model_name,
        output_dir=args.output_dir,
        add_sampling_metadata=args.add_sampling_metadata,
        test_FLOPs=args.test_FLOPs,
        monitor_gpu_usage=args.monitor_gpu_usage,
        interval=args.interval,
        max_order=args.max_order,
        first_enhance=args.first_enhance,
    )

    main(opts)
    # CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 sample_ddp.py