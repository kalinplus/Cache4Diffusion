import argparse
import os
import time
import torch
from inference_utils import get_torch_dtype, setup_pipeline, sanitize_filename

# Maps model_name -> the guidance scale kwarg accepted by that pipeline
_GUIDANCE_KWARG = {
    "flux": "guidance_scale",
    "qwen_image": "true_cfg_scale",
    "hunyuan_video": "guidance_scale",
}

# Models that require device_map='cuda' loading instead of .to(device)
_USE_DEVICE_MAP = {"qwen_image"}

# Video models: output .frames instead of .images, and accept video-specific kwargs
_VIDEO_MODELS = {"hunyuan_video"}

# Default negative prompts per model (can be overridden via --negative_prompt)
_DEFAULT_NEGATIVE_PROMPT = {
    "qwen_image": "blurry, low resolution, bad anatomy, watermark",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-prompt inference with TaylorSeer caching")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--prefix", type=str, default="TaylorSeer")
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--strategy", type=str, default="taylorseer", choices=["taylorseer"])
    parser.add_argument("--model_name", type=str, default="flux",
                        help="Adapter name: 'flux', 'qwen_image', 'hunyuan_video', ...")
    parser.add_argument("--enable_cpu_offload", action="store_true")
    # Video-specific args (ignored for image models)
    parser.add_argument("--video_length", type=int, default=None,
                        help="Number of frames (video models only)")
    parser.add_argument("--video_size", type=int, nargs=2, default=None,
                        metavar=("HEIGHT", "WIDTH"),
                        help="Video resolution as HEIGHT WIDTH (video models only)")
    parser.add_argument("--fps", type=int, default=None,
                        help="Frames per second for saved video (video models only)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    torch_dtype = get_torch_dtype(args.dtype)
    if args.device == "cpu" and torch_dtype in (torch.float16, torch.bfloat16):
        print("Requested low-precision dtype on CPU; overriding to float32.")
        torch_dtype = torch.float32

    if args.enable_cpu_offload:
        raise NotImplementedError("CPU offload is not supported for TaylorSeer yet.")

    use_device_map = args.model_name in _USE_DEVICE_MAP
    pipeline = setup_pipeline(args.model, int(args.steps), args.strategy, args.model_name,
                              torch_dtype, args.device, use_device_map=use_device_map)

    is_cuda = args.device == "cuda" and torch.cuda.is_available()
    if is_cuda:
        parameter_peak_memory = torch.cuda.max_memory_allocated(device="cuda")
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        start_time = time.time()

    guidance_kwarg = _GUIDANCE_KWARG.get(args.model_name, "guidance_scale")
    call_kwargs = {
        "num_inference_steps": int(args.steps),
        "generator": torch.Generator("cpu").manual_seed(int(args.seed)),
        guidance_kwarg: float(args.guidance_scale),
    }
    if args.negative_prompt:
        call_kwargs["negative_prompt"] = args.negative_prompt
    elif args.model_name in _DEFAULT_NEGATIVE_PROMPT:
        call_kwargs["negative_prompt"] = _DEFAULT_NEGATIVE_PROMPT[args.model_name]

    is_video = args.model_name in _VIDEO_MODELS
    if is_video:
        if args.video_length is not None:
            call_kwargs["num_frames"] = args.video_length
        if args.video_size is not None:
            call_kwargs["height"], call_kwargs["width"] = args.video_size

    result = pipeline(args.prompt, **call_kwargs)

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
    if is_video:
        from diffusers.utils import export_to_video
        frames = result.frames[0]  # list of PIL images
        fps = args.fps or 8
        save_path = os.path.join(args.outdir, f"{args.prefix}_{safe}.mp4")
        export_to_video(frames, save_path, fps=fps)
    else:
        save_path = os.path.join(args.outdir, f"{args.prefix}_{safe}.png")
        result.images[0].save(save_path)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
