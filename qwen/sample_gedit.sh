export CUDA_VISIBLE_DEVICES=1
torchrun --standalone --nproc_per_node=1 taylorseer/sample_gedit.py \
    --dataset_path "/mnt/data0/datasets/stepfun-ai/GEdit-Bench" \
    --model_name "/mnt/data0/Qwen-Image-Edit-2509" \
    # --english_only
    # --test_FLOPs \
    # --monitor_gpu_usage

# export CUDA_VISIBLE_DEVICES=2
# torchrun --standalone --nproc_per_node=1 qwen/sample_gedit.py \
#     --dataset_path "/mnt/data0/datasets/stepfun-ai/GEdit-Bench" \
#     --output_dir "samples/Gedit/baseline" \
#     --model_name "/mnt/data0/Qwen-Image-Edit-2509" \
#     # --english_only