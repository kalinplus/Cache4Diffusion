import torch
import argparse
import time
import re
import os
from hyimage.diffusion.pipelines.hunyuanimage_pipeline import HunyuanImagePipeline
import loguru

def sanitize_filename(text: str, max_length: int = 100) -> str:
    """Clean and sanitize text for use as filename."""
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("/", "-")
    text = re.sub(r"[^\w\-\\s]", "", text)
    text = text.replace(" ", "_")
    if len(text) == 0:
        text = "prompt"
    return text[:max_length]

def load_prompts_from_file(prompt_file: str) -> list:
    """Load prompts from a text file (one prompt per line)."""
    prompts = []
    with open(prompt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                prompts.append(line)
    return prompts

def parse_args():
    parser = argparse.ArgumentParser(description='Batch generate images using HunyuanImage model')

    # Prompt source: either single prompt or prompt file
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prompt', type=str,
                       help='Single text prompt for image generation')
    group.add_argument('--prompt_file', type=str,
                       help='Path to text file containing prompts (one per line)')

    parser.add_argument('--model_name', type=str,
                       default="hunyuanimage-v2.1",
                       choices=["hunyuanimage-v2.1", "hunyuanimage-v2.1-distilled"],
                       help='Model name to use (default: hunyuanimage-v2.1)')

    parser.add_argument('--use_reprompt', action='store_true', default=False,
                       help='Enable prompt enhancement (default: False)')

    parser.add_argument('--use_refiner', action='store_true', default=False,
                       help='Enable refiner model (default: False)')

    parser.add_argument('--shift', type=float, default=5.0,
                       help='Shift parameter (default: 5.0)')

    parser.add_argument('--seed', type=int, default=649151,
                       help='Random seed for generation (default: 649151)')

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

    # Batch processing options
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Start index for batch processing (default: 0)')

    parser.add_argument('--end_idx', type=int, default=None,
                       help='End index for batch processing (default: process all)')

    parser.add_argument('--seed_offset', type=int, default=None,
                       help='Add offset to seed for each prompt (default: use same seed)')

    return parser.parse_args()

def main():
    args = parse_args()

    # Get prompts
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = load_prompts_from_file(args.prompt_file)

    # Apply start/end index filtering
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(prompts)
    prompts = prompts[start_idx:end_idx]

    total_prompts = len(prompts)
    loguru.logger.info(f"Total prompts to process: {total_prompts}")

    # Create output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)

    # Load model once
    loguru.logger.info(f"Loading model: {args.model_name}")
    pipe = HunyuanImagePipeline.from_pretrained(model_name=args.model_name, torch_dtype='bf16')
    pipe = pipe.to("cuda")

    # Determine number of inference steps based on model type
    num_inference_steps = 8 if "distilled" in args.model_name else 50

    # Log configuration
    loguru.logger.info(f"Configuration: reprompt={args.use_reprompt}, refiner={args.use_refiner}, shift={args.shift}")

    # Process each prompt
    for idx, prompt in enumerate(prompts):
        actual_idx = start_idx + idx
        current_seed = args.seed + (actual_idx if args.seed_offset is not None else 0)

        loguru.logger.info(f"\n[{actual_idx+1}/{total_prompts}] Processing prompt: {prompt[:100]}...")
        loguru.logger.info(f"Seed: {current_seed}")

        # Memory tracking setup
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start_time = time.time()

        try:
            image = pipe(
                prompt=prompt,
                # Examples of supported resolutions and aspect ratios for HunyuanImage-2.1:
                # 16:9  -> width=2560, height=1536
                # 4:3   -> width=2304, height=1792
                # 1:1   -> width=2048, height=2048
                # 3:4   -> width=1792, height=2304
                # 9:16  -> width=1536, height=2560
                # Please use one of the above width/height pairs for best results.
                width=args.width,
                height=args.height,
                use_reprompt=args.use_reprompt,
                use_refiner=args.use_refiner,
                num_inference_steps=num_inference_steps,
                guidance_scale=args.guidance_scale,
                shift=args.shift,
                seed=current_seed,
            )

            # Performance measurement and reporting
            if is_cuda:
                end.record()
                torch.cuda.synchronize()
                elapsed_time = start.elapsed_time(end) * 1e-3
                peak_memory = torch.cuda.max_memory_allocated(device="cuda")
                loguru.logger.info(
                    f"Performance stats - Time: {elapsed_time:.2f}s, Peak memory: {peak_memory/1e9:.2f}GB"
                )
            else:
                elapsed_time = time.time() - start_time
                loguru.logger.info(f"Performance stats - Time: {elapsed_time:.2f}s")

            # Generate filename with index
            safe_prompt = sanitize_filename(prompt, max_length=80)

            # Build filename with index and optional prefix
            if args.prefix:
                filename = f"{args.prefix}_{actual_idx:04d}_{safe_prompt}.png"
            else:
                filename = f"{actual_idx:04d}_{safe_prompt}.png"

            output_path = os.path.join(args.outdir, filename)

            loguru.logger.info(f"Saving image to: {output_path}")
            image.save(output_path)
            loguru.logger.info(f"[{actual_idx+1}/{total_prompts}] Completed successfully")

        except Exception as e:
            loguru.logger.error(f"[{actual_idx+1}/{total_prompts}] Failed: {e}")
            continue

    loguru.logger.info(f"\nBatch processing completed! Processed {total_prompts} prompts.")

if __name__ == "__main__":
    main()
