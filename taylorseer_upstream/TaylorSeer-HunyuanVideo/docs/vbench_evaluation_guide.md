# VBench Evaluation Guide

## What is VBench

VBench is a comprehensive benchmark suite for evaluating video generation models ([CVPR 2024 Highlight](https://arxiv.org/abs/2311.17982), [GitHub](https://github.com/Vchitect/VBench)). It decomposes video quality into **16 evaluation dimensions** organized into **Quality** and **Semantic** categories, and uses classical vision models (CLIP, DINO, RAFT, object detectors) to compute quantitative scores — no LLM involved.

## Evaluation Pipeline

```
Step 1: sample_video_vbench.py  ->  generate videos from VBench prompts
Step 2: calc_vbench.py          ->  run VBench library evaluation per dimension
Step 3: tabulate_vbench_scores.py -> normalize scores, compute quality/semantic/total
```

## Step 1: Generate Videos from VBench Prompts

### Prompts File

The prompt dataset is `eval/VBench_full_info.json`, containing hundreds of prompts each tagged with which dimension(s) it evaluates:

```json
[
    {"prompt_en": "In a still frame, a stop sign", "dimension": ["temporal_flickering"]},
    {"prompt_en": "a toilet, frozen in time", "dimension": ["temporal_flickering"]},
    ...
]
```

### Using the Shell Script (Recommended)

The shell script `eval/sample_vbench.sh` handles multi-GPU splitting automatically:

```bash
# Usage:
# ./eval/sample_vbench.sh <full_info_path> <Num_Devices> <SEED> <Num_Samples> <Video_Save_Path> <Path2Log>

# Single GPU example:
./eval/sample_vbench.sh ./eval/ 1 42 5 ./results/vbench ./logs/vbench/

# Multi-GPU example (4 GPUs):
./eval/sample_vbench.sh ./eval/ 4 42 5 ./results/vbench ./logs/vbench/
```

The script splits prompts across GPUs using `CUDA_VISIBLE_DEVICES` and `--index-start`/`--index-end`.

### Direct CLI Invocation

```bash
CUDA_VISIBLE_DEVICES=0 python sample_video_vbench.py \
    --vbench-json-path ./eval/VBench_full_info.json \
    --index-start 0 --index-end 99 \
    --seed 42 --num-videos-per-prompt 5 \
    --video-size 480 640 --video-length 65 \
    --infer-steps 50 --flow-reverse --use-cpu-offload \
    --save-path ./results/vbench
```

### Key CLI Arguments for VBench

| Argument | Default | VBench Recommended |
|----------|---------|-------------------|
| `--vbench-json-path` | None | **Required** — path to VBench JSON prompt file |
| `--num-videos-per-prompt` | 5 | 5 (VBench standard) |
| `--index-start` / `--index-end` | 0 / 0 | Prompt range (auto-set by shell script) |
| `--seed` | None | 42 (or any fixed seed for reproducibility) |
| `--video-size` | 720 1280 | **480 640** (VBench standard, saves GPU memory) |
| `--video-length` | 129 | **65** (VBench standard) |
| `--infer-steps` | 50 | 50 |
| `--flow-reverse` | false | **Required** |
| `--use-cpu-offload` | false | Recommended if GPU memory is tight |
| `--save-path` | ./results | Directory for saved `.mp4` files |
| `--model-base` | ckpts | Root path of model weights |

### Resume Behavior

The script checks `if os.path.exists(cur_save_path)` and skips already-generated videos, so you can safely resume interrupted runs without regenerating existing videos.

## Step 2: Evaluate with VBench Metrics

```bash
python eval/vbench/calc_vbench.py \
    ./results/vbench \
    ./results/vbench_scores \
    --start 0 --end -1
```

Arguments:
- `video_folder` — directory containing generated `.mp4` videos
- `save_path` — directory to save evaluation results (`*_eval_results.json` per dimension)
- `--start` / `--end` — index range of dimensions to evaluate (default: all)

### How VBench Scores Work

VBench uses three core metric primitives:

| Primitive | Model | Measures |
|-----------|-------|----------|
| **CLIP similarity** | ViT-L/14 | Text-image alignment, visual consistency |
| **DINO feature distance** | DINO ViT | Perceptual change between frames |
| **Optical flow** | RAFT | Motion magnitude and smoothness |

#### Quality Dimensions

- **`temporal_flickering`** — DINO feature distance between consecutive frames (lower = less flickering)
- **`motion_smoothness`** — RAFT optical flow warping error (lower = smoother)
- **`dynamic_degree`** — RAFT optical flow magnitude (higher = more motion)
- **`subject_consistency`** — CLIP visual feature similarity of subject region across frames (higher = more consistent)
- **`background_consistency`** — CLIP visual feature similarity of background across frames (higher = more consistent)
- **`imaging_quality`** — CLIP-IQA image quality score
- **`aesthetic_quality`** — CLIP-based aesthetic scoring

#### Semantic Dimensions

- **`object_class`** — CLIP text-image similarity on object class description
- **`human_action`** — CLIP text-image similarity on action description
- **`scene`** — CLIP text-image similarity on scene description
- **`color`** — CLIP text-image similarity on color attributes
- **`appearance_style`** — CLIP text-image similarity on style attributes
- **`temporal_style`** — CLIP style feature similarity across frames
- **`multiple_objects`** — Object detection count + CLIP class verification
- **`spatial_relationship`** — Detected object positions vs prompt description
- **`overall_consistency`** — Overall frame-to-frame consistency

## Step 3: Tabulate Final Scores

```bash
python eval/vbench/tabulate_vbench_scores.py --score_dir ./results/vbench_scores/vbench
```

This reads all `*_eval_results.json` files, normalizes each dimension to [0,1] using predefined min/max ranges, and computes aggregate scores:

- **Quality score** (weight 4): subject/background consistency, temporal flickering, motion smoothness, dynamic degree, aesthetic quality, imaging quality
- **Semantic score** (weight 1): object class, multiple objects, human action, color, spatial relationship, scene, appearance style, temporal style, overall consistency
- **Total score**: `(quality * 4 + semantic * 1) / 5`

Output files:
- `all_results.json` — raw per-dimension scores
- `scaled_results.json` — normalized and weighted scores in percentage

## File Structure

```
eval/
  VBench_full_info.json          # VBench prompt dataset
  sample_vbench.sh               # Multi-GPU video generation script
  vbench/
    calc_vbench.py               # Run VBench metric evaluation
    tabulate_vbench_scores.py    # Normalize and aggregate scores
```
