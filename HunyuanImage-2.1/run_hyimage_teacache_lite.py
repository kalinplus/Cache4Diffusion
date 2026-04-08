import torch
import argparse
import time
import re
import os
from hyimage.diffusion.pipelines.hunyuanimage_pipeline import HunyuanImagePipeline
import loguru
from scripts.teacache_lite_hyimage.forwards.apply_teacache_lite_hyimage_pipeline import apply_teacache_lite_hyimage_pipeline
from scripts.teacache_lite_hyimage.forwards.apply_teacache_lite_hyimage_forward import apply_teacache_lite_hyimage_forward


def sanitize_filename(text: str, max_length: int = 100) -> str:
    """Clean and sanitize text for use as filename."""
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("/", "-")
    text = re.sub(r"[^\w\-\\s]", "", text)
    text = text.replace(" ", "_")
    if len(text) == 0:
        text = "prompt"
    return text[:max_length]


def parse_args():
    parser = argparse.ArgumentParser(description='Single image generation with TeaCache-Lite for HunyuanImage')

    parser.add_argument('--prompt', type=str, required=True,
                       help='Single text prompt for image generation')

    parser.add_argument('--model_name', type=str,
                       default="hunyuanimage-v2.1",
                       help='Model name to use (default: hunyuanimage-v2.1)')

    parser.add_argument('--use_reprompt', action='store_true', default=False,
                       help='Enable prompt enhancement (default: False)')

    parser.add_argument('--use_refiner', action='store_true', default=False,
                       help='Enable refiner model (default: False)')

    parser.add_argument('--seed', type=int, default=649151,
                       help='Random seed for generation (default: 649151)')

    parser.add_argument('--shift', type=float, default=5.0,
                       help='Shift parameter (default: 5.0)')

    parser.add_argument('--width', type=int, default=2048,
                       help='Image width (default: 2048)')

    parser.add_argument('--height', type=int, default=2048,
                       help='Image height (default: 2048)')

    parser.add_argument('--guidance_scale', type=float, default=3.5,
                       help='Guidance scale (default: 3.5)')

    parser.add_argument('--outdir', type=str, default="outputs",
                       help='Directory to save the image (default: outputs)')

    parser.add_argument('--prefix', type=str, default="",
                       help='Filename prefix for the image (default: no prefix)')

    parser.add_argument('--rel_l1_thresh', type=float, default=0.6,
                       help='TeaCache relative L1 threshold (lambda) (default: 0.6)')

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    loguru.logger.info(f"Loading model: {args.model_name}")
    pipe = HunyuanImagePipeline.from_pretrained(model_name=args.model_name, torch_dtype='bf16')
    pipe = pipe.to("cuda")
    apply_teacache_lite_hyimage_pipeline(pipe, rel_l1_thresh=args.rel_l1_thresh)
    apply_teacache_lite_hyimage_forward(pipe.dit)

    num_inference_steps = 8 if "distilled" in args.model_name else 50

    loguru.logger.info(f"Configuration: reprompt={args.use_reprompt}, refiner={args.use_refiner}, shift={args.shift}")
    loguru.logger.info(f"TeaCache: rel_l1_thresh={args.rel_l1_thresh}")

    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    image = pipe(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        use_reprompt=args.use_reprompt,
        use_refiner=args.use_refiner,
        num_inference_steps=num_inference_steps,
        guidance_scale=args.guidance_scale,
        shift=args.shift,
        seed=args.seed,
    )

    end.record()
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end) * 1e-3
    peak_memory = torch.cuda.max_memory_allocated(device="cuda")
    loguru.logger.info(
        f"Performance stats - Time: {elapsed_time:.2f}s, Peak memory: {peak_memory/1e9:.2f}GB"
    )

    safe_prompt = sanitize_filename(args.prompt, max_length=80)
    if args.prefix:
        filename = f"{args.prefix}_{safe_prompt}.png"
    else:
        filename = f"{safe_prompt}.png"

    output_path = os.path.join(args.outdir, filename)
    loguru.logger.info(f"Saving image to: {output_path}")
    image.save(output_path)
    loguru.logger.info("Completed successfully")


if __name__ == "__main__":
    main()
