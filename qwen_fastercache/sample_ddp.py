"""
DDP sampling script for Qwen image generation with FasterCache acceleration.

Usage example (single node, 4 GPUs):
    torchrun --nproc_per_node=4 sample_ddp.py \
        --model_name qwen-image \
        --prompt_file prompts/abc.txt \
        --output_dir samples/fastercache_test \
        --num_steps 50 \
        --fc_start_step 15 \
        --fc_interval 2 \
        --fc_alpha 0.3

FasterCache vs freqca differences
-----------------------------------
- No --interval / --max_order / --forecast_method / --decompose_method args.
- Three new args: --fc_start_step, --fc_interval, --fc_alpha.
- No cache_init() / cal_type() / cache_step() calls.
- The transformer forward receives a plain integer `counter` (step index).
"""

import os
import torch
from PIL import Image, ExifTags
from tqdm import tqdm
from dataclasses import dataclass
from transformers import pipeline as hf_pipeline
import torch.distributed as dist

from pipeline.pipeline_qwenimage import QwenImagePipeline
from fastercache_utils import pipeline_with_fastercache

NSFW_THRESHOLD = 0.85


@dataclass
class SamplingOptions:
    prompts: list[str]
    negative_prompt: str
    width: int
    height: int
    num_steps: int
    guidance_scale: float
    true_cfg_scale: float
    seed: int | None
    num_images_per_prompt: int
    batch_size: int
    model_name: str
    model_path: str | None      # explicit model path; overrides model_name lookup if set
    output_dir: str
    add_sampling_metadata: bool
    use_nsfw_filter: bool
    test_FLOPs: bool
    # FasterCache schedule
    fc_start_step: int
    fc_interval: int
    fc_alpha: float


def main(opts: SamplingOptions):
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    device     = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    # Distribute prompts across ranks
    total_prompts = len(opts.prompts)
    per_proc  = (total_prompts + world_size - 1) // world_size
    start     = rank * per_proc
    end       = min(start + per_proc, total_prompts)
    local_prompts = opts.prompts[start:end]

    if rank == 0:
        os.makedirs(opts.output_dir, exist_ok=True)

    # Optional NSFW classifier
    nsfw_classifier = None
    if opts.use_nsfw_filter:
        nsfw_classifier = hf_pipeline(
            "image-classification",
            model="Falconsai/nsfw_image_detection",
            device=device,
        )

    # Load pipeline
    model_path = opts.model_path if opts.model_path else {
        "qwen-image":      "/apdcephfs_jn/share_302243908/jiachengliu/Qwen/Qwen-Image",
        "qwen-image-edit": "/apdcephfs_jn/share_302243908/jiachengliu/Qwen/Qwen-Image-Edit",
    }.get(opts.model_name)
    if model_path is None:
        raise ValueError(f"Unsupported model_name: {opts.model_name}. Pass --model_path to specify explicitly.")

    pipe = QwenImagePipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).to(device=device)

    # Enable FasterCache
    pipe = pipeline_with_fastercache(
        pipe,
        start_step=opts.fc_start_step,
        cache_interval=opts.fc_interval,
        alpha=opts.fc_alpha,
    )

    # Wire FLOPs flag into the pipeline so the denoising loop can read it
    pipe._test_FLOPs = opts.test_FLOPs

    if rank == 0:
        print(
            f"[FasterCache] start_step={opts.fc_start_step}, "
            f"interval={opts.fc_interval}, alpha={opts.fc_alpha}"
        )

    local_images    = len(local_prompts) * opts.num_images_per_prompt
    progress_bar    = tqdm(total=local_images, desc=f"[rank {rank}] Generating")
    num_batches     = (len(local_prompts) + opts.batch_size - 1) // opts.batch_size

    for batch_idx in range(num_batches):
        prompt_start = batch_idx * opts.batch_size
        prompt_end   = min(prompt_start + opts.batch_size, len(local_prompts))
        batch_prompts = local_prompts[prompt_start:prompt_end]
        n_prompts     = len(batch_prompts)

        for image_idx in range(opts.num_images_per_prompt):
            generators = []
            for i in range(n_prompts):
                global_prompt_idx = start + prompt_start + i
                global_img_idx    = global_prompt_idx * opts.num_images_per_prompt + image_idx
                seed = (opts.seed + global_img_idx) if opts.seed is not None \
                       else torch.randint(0, 2**32, (1,)).item()
                generators.append(torch.Generator(device).manual_seed(int(seed)))

            result = pipe(
                prompt=batch_prompts,
                negative_prompt=opts.negative_prompt,
                height=opts.height,
                width=opts.width,
                num_inference_steps=opts.num_steps,
                guidance_scale=opts.guidance_scale,
                true_cfg_scale=opts.true_cfg_scale,
                generator=generators,
                # FasterCache params (pipeline resolves defaults from transformer attrs)
                fastercache_start_step=opts.fc_start_step,
                fastercache_interval=opts.fc_interval,
                fastercache_alpha=opts.fc_alpha,
            )

            images = getattr(result, "images", None)
            if images is None:
                images = list(result) if isinstance(result, (list, tuple)) else [result]

            for i, img in enumerate(images):
                if not isinstance(img, Image.Image):
                    continue

                # NSFW filter
                nsfw_score = 0.0
                if opts.use_nsfw_filter and nsfw_classifier is not None:
                    nsfw_result = nsfw_classifier(img)
                    nsfw_score  = next(
                        (r["score"] for r in nsfw_result if r["label"] == "nsfw"), 0.0
                    )

                if nsfw_score < NSFW_THRESHOLD:
                    exif_data = Image.Exif()
                    exif_data[ExifTags.Base.Software] = "AI generated;t2i;qwen_fastercache"
                    exif_data[ExifTags.Base.Make]     = "Qwen"
                    exif_data[ExifTags.Base.Model]    = opts.model_name
                    if opts.add_sampling_metadata and i < len(batch_prompts):
                        exif_data[ExifTags.Base.ImageDescription] = batch_prompts[i]

                    global_prompt_idx = start + prompt_start + i
                    global_img_idx    = global_prompt_idx * opts.num_images_per_prompt + image_idx
                    filename = f"{opts.output_dir}/img_{global_img_idx}.jpg"
                    img.save(filename, exif=exif_data, quality=95, subsampling=0)
                else:
                    print(f"[rank {rank}] Skipped image (NSFW score={nsfw_score:.3f})")

                if rank == 0 and progress_bar is not None:
                    progress_bar.update(1)

    if rank == 0 and progress_bar is not None:
        progress_bar.close()

    dist.barrier()
    if rank == 0:
        print("All images generated.")
    dist.destroy_process_group()


def read_prompts(prompt_file: str):
    with open(prompt_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen image generation with FasterCache (DDP).")

    # Model / I/O
    parser.add_argument("--model_name",    type=str, default="qwen-image",
                        choices=["qwen-image", "qwen-image-edit"])
    parser.add_argument("--model_path",    type=str, default=None,
                        help="Explicit path to the pretrained model directory. "
                             "Overrides the default path looked up from --model_name.")
    parser.add_argument("--prompt_file",   type=str,
                        default="/apdcephfs_cq11/share_300483685/jiachengliu/code/qwen_final/taylorseer/prompts/abc.txt")
    parser.add_argument("--negative_prompt", type=str, default=" ")
    parser.add_argument("--output_dir",    type=str, default="samples/fastercache_test")

    # Generation
    parser.add_argument("--width",  type=int,   default=1328)
    parser.add_argument("--height", type=int,   default=1328)
    parser.add_argument("--num_steps",    type=int,   default=50)
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="Guidance scale for guidance-distilled models.")
    parser.add_argument("--true_cfg_scale", type=float, default=1.0,
                        help="True CFG scale. Set >1 with --negative_prompt to enable CFG.")
    parser.add_argument("--seed",   type=int,   default=0)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--batch_size",  type=int, default=1,
                        help="Number of prompts processed in a single forward pass.")
    parser.add_argument("--add_sampling_metadata", action="store_true")
    parser.add_argument("--use_nsfw_filter",        action="store_true")
    parser.add_argument("--test_FLOPs",             action="store_true",
                        help="Measure per-step FLOPs with calflops and print speedup ratio. "
                             "Only processes the first prompt to keep measurement fast.")

    # FasterCache schedule
    parser.add_argument("--fc_start_step", type=int,   default=15,
                        help="Steps 0..fc_start_step always run full attention (warm-up). "
                             "Default 15 for a 50-step schedule.")
    parser.add_argument("--fc_interval",   type=int,   default=2,
                        help="Attention is recomputed every fc_interval steps; "
                             "others reuse cache.  2 → ~50%% saved, 3 → ~67%% saved.")
    parser.add_argument("--fc_alpha",      type=float, default=0.3,
                        help="Linear extrapolation coefficient for skip steps. "
                             "0.0 = plain reuse, 0.3 = mild extrapolation.")

    args = parser.parse_args()
    prompts = read_prompts(args.prompt_file)

    opts = SamplingOptions(
        prompts=prompts,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        true_cfg_scale=args.true_cfg_scale,
        seed=args.seed,
        num_images_per_prompt=args.num_images_per_prompt,
        batch_size=args.batch_size,
        model_name=args.model_name,
        model_path=args.model_path,
        output_dir=args.output_dir,
        add_sampling_metadata=args.add_sampling_metadata,
        use_nsfw_filter=args.use_nsfw_filter,
        test_FLOPs=args.test_FLOPs,
        fc_start_step=args.fc_start_step,
        fc_interval=args.fc_interval,
        fc_alpha=args.fc_alpha,
    )

    main(opts)
