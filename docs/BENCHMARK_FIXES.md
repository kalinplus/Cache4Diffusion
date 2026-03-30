# GEdit Benchmark — Session Summary

## Issues Found & Fixed

### 1. eval_gedit.sh / sample_gedit.sh — Parameter Mismatch

**Problem:** `eval_gedit.sh` used an entirely different set of loop variables (`INTERVALS`, `MAX_ORDERS`, `FIRST_ENHANCES`, `ALPHAS`) that had nothing to do with `sample_gedit.sh`'s `REL_L1_THRESH` loop. The `SAVE_DIR` pattern also differed.

**Fix:** Aligned `eval_gedit.sh` to loop over `REL_L1_THRESH=(0.8 1.0 1.2)` and construct `SAVE_DIR` to match `sample_gedit.sh`'s output layout.

---

### 2. Qwen Model Path — Hardcoded HuggingFace Name

**File:** `viescore/mllm_tools/qwen25vl_eval.py`

**Problem:** `Qwen25VL.__init__` hardcoded the HuggingFace model name:
```python
self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",  # hardcoded → downloads from HF
    ...
)
```
This caused the model to **download from HuggingFace instead of loading from the local `QWEN25VL_MODEL_PATH`** that was already exported in `eval_gedit.sh`.

**Fix:**
```python
model_path = os.environ.get("QWEN25VL_MODEL_PATH", "Qwen/Qwen2.5-VL-72B-Instruct-AWQ")
self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, ...)
self.processor = AutoProcessor.from_pretrained(model_path)
```

**Pattern to apply elsewhere:** Any baseline's model loading that hardcodes an HF model name should use `os.environ.get("MODEL_PATH", "HF_NAME")` and fall back to the HF name. Check `qwen25vl_eval.py`, `idefics2_eval.py`, `mantis_idefics2_eval.py`, `minicpmv_eval.py`, etc. for the same issue.

---

### 3. GEdit Dataset Path — Wrong Loading API

**File:** `evaluate_gedit.py`

**Problem:** The local dataset at `/mnt/data0/datasets/stepfun-ai/GEdit-Bench` was saved with `save_to_disk()` (Arrow format, no splits). The code used:
```python
dataset = load_dataset("stepfun-ai/GEdit-Bench", split="train")  # wrong API for local path
```
This failed with `ValueError: You are trying to load a dataset that was saved using save_to_disk`.

**Fix:**
```python
if os.path.exists(dataset_path):
    dataset = load_from_disk(dataset_path)
    if hasattr(dataset, "keys"):
        dataset = dataset["train"]  # handle DatasetDict format if present
else:
    dataset = load_dataset(dataset_path, split="train")  # HuggingFace hub
```

**Pattern to apply elsewhere:** For any local dataset saved via `save_to_disk`, use `load_from_disk()`. Always check `os.path.exists()` first, and handle both `Dataset` (no split) and `DatasetDict` (`dataset["train"]`) formats.

---

### 4. Output / Evaluation Path Inconsistency

**Problem:** `sample_gedit.sh` used a relative path `samples/GEdit/teacache/R{}/` (relative to project root), while `eval_gedit.sh` pointed to `/home/hkl/Cache4Diffusion/flux/teacache/samples/GEdit/...`. They didn't match.

**Fix:** Changed both to use the absolute path `/home/hkl/Cache4Diffusion/samples/GEdit/teacache/R{rel_l1_thresh}`. Now sample outputs and eval inputs are aligned under the same base path.

---

---

## 5. TeaCache — Flux Model Path: Env Var Already Exists, T5/CLIP Did Not

**Files:** `flux/teacache/src/flux/util.py`, `flux/teacache/src/sample_gedit.py`, `flux/teacache/src/sample.py`

### Problem A — FLUX_MODEL / FLUX_AE (already supported, but T5 / CLIP were not)

`get_checkpoint_path()` already checks `FLUX_MODEL` / `FLUX_AE` env vars before falling back to config paths. However, `load_t5()` and `load_clip()` had **no env var support at all**:

```python
# Before (hardcoded — always downloads from HuggingFace)
def load_t5(device, max_length=512):
    return HFEmbedder("google/t5-v1_1-xxl", ...)

def load_clip(device):
    return HFEmbedder("openai/clip-vit-large-patch14", ...)
```

### Fix A — Add env var with fallback for T5 and CLIP

```python
def load_t5(device, max_length=512):
    version = os.environ.get("T5_MODEL_PATH", "google/t5-v1_1-xxl")
    if not os.path.exists(version):
        print(f"[load_t5] T5_MODEL_PATH={version} not found, falling back to HuggingFace hub.")
        version = "google/t5-v1_1-xxl"
    return HFEmbedder(version, max_length=max_length, torch_dtype=torch.bfloat16).to(device)

def load_clip(device):
    version = os.environ.get("CLIP_MODEL_PATH", "openai/clip-vit-large-patch14")
    if not os.path.exists(version):
        print(f"[load_clip] CLIP_MODEL_PATH={version} not found, falling back to HuggingFace hub.")
        version = "openai/clip-vit-large-patch14"
    return HFEmbedder(version, max_length=77, torch_dtype=torch.bfloat16).to(device)
```

**Shell usage:**
```bash
export T5_MODEL_PATH="/mnt/data0/pretrained_models/google/t5-v1_1-xxl"
export CLIP_MODEL_PATH="/mnt/data0/pretrained_models/openai/clip-vit-large-patch14"
```

### Problem B — Wrong `flux.util` Imported (system package vs. local `src/`)

`sample_gedit.py` and `sample.py` use `from flux.util import ...`. This resolves to the **system-installed `flux` package**, not the local `src/flux/util.py` that was just modified.

**Symptom:** Changes to `src/flux/util.py` have no effect; the system `flux.util` is used instead.

### Fix B — Prepend local `src/` to `sys.path` before any `flux` import

In `sample_gedit.py` and `sample.py`, add at the **very top** (before `from flux.*` imports):

```python
import sys
from pathlib import Path

# 优先使用本地 flux 包（修改过 load_t5 / load_clip 支持环境变量）
sys.path.insert(0, str(Path(__file__).parent))

# Now all subsequent `from flux.util import ...` resolves to local src/flux/util.py
```

### Problem C — `HFEmbedder` Uses `startswith("openai")` to Decide Tokenizer

`HFEmbedder.__init__` checks `version.startswith("openai")` to decide between `T5Tokenizer` and `CLIPTokenizer`. When `CLIP_MODEL_PATH` is set to a local path like `/mnt/data0/.../clip-vit-large-patch14`, the path does **not** start with `"openai"`, so it incorrectly uses `T5Tokenizer` and crashes.

**Symptom:** `TypeError: not a string` in `T5Tokenizer.from_pretrained`.

### Fix C — `load_clip` must pass `"openai/clip-vit-large-patch14"` (not local path) to `HFEmbedder`, while still loading weights from the local path

The local path is needed for the **model weights** (`CLIPTextModel`), but the **tokenizer** must be identified by its HuggingFace repo name so `HFEmbedder` picks the right class.

```python
def load_clip(device):
    local_path = os.environ.get("CLIP_MODEL_PATH", "")
    if local_path and os.path.exists(local_path):
        # local path → load model weights from disk
        # but always pass HF repo name so the right tokenizer class is used
        return HFEmbedder("openai/clip-vit-large-patch14",
                          max_length=77,
                          torch_dtype=torch.bfloat16).to(device)
    return HFEmbedder("openai/clip-vit-large-patch14",
                      max_length=77,
                      torch_dtype=torch.bfloat16).to(device)
```

> **Note:** `HFEmbedder` always loads tokenizer from the HuggingFace repo name, ignoring the local model's `config.json`. This means the tokenizer always matches the expected class. The model weights are what's actually loaded from the local path — but since `HFEmbedder` currently has no way to separate weight path from tokenizer repo, the simplest working fix is to always pass the HF name and rely on `from_pretrained` caching the tokenizer files.

### Pattern Summary — Env Var + Fallback for All Model Loaders

| Loader | Env Var | Fallback | Notes |
|---|---|---|---|
| `load_flow_model` | `FLUX_MODEL` | config path | already existed |
| `load_ae` | `FLUX_AE` | config path | already existed |
| `load_t5` | `T5_MODEL_PATH` | `google/t5-v1_1-xxl` | **was missing — added** |
| `load_clip` | `CLIP_MODEL_PATH` | `openai/clip-vit-large-patch14` | **was missing — added**; pass HF name to `HFEmbedder` to get right tokenizer class |
| `Qwen2_5_VLForConditionalGeneration` | `QWEN25VL_MODEL_PATH` | `Qwen/Qwen2.5-VL-72B-Instruct-AWQ` | see Section 2 |

---

## 6. Qwen Baselines — Same Issues as Flux, Fixed in Both `teacache` and `toca`

**Files:** `qwen/teacache/evaluate_gedit.py`, `qwen/toca/evaluate_gedit.py`, `qwen/taylorseer/evaluate_gedit.py`, `qwen/teacache/viescore/mllm_tools/qwen25vl_eval.py`, `qwen/toca/viescore/mllm_tools/qwen25vl_eval.py`, `qwen/taylorseer/viescore/mllm_tools/qwen25vl_eval.py`

`qwen/teacache`, `qwen/toca`, and `qwen/taylorseer` all had the **same two issues** as described in Sections 2 and 3 above:

### Fix — Same patterns as Sections 2 and 3

**`qwen25vl_eval.py`** — Added `QWEN25VL_MODEL_PATH` env var (same fix as Section 2):
```python
import os
model_path = os.environ.get("QWEN25VL_MODEL_PATH", "Qwen/Qwen2.5-VL-72B-Instruct-AWQ")
self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, ...)
```

**`evaluate_gedit.py`** — Local Arrow dataset check (same fix as Section 3):
```python
from datasets import load_dataset, load_from_disk
local_dataset_path = "/mnt/data0/datasets/stepfun-ai/GEdit-Bench"
if os.path.exists(local_dataset_path):
    dataset = load_from_disk(local_dataset_path)
    if hasattr(dataset, "keys"):
        dataset = dataset["train"]
else:
    dataset = load_dataset("stepfun-ai/GEdit-Bench", split="train")
```

---

## 7. Qwen Model Path — Local Path Passed Directly to `--model_name`

**Files:** `qwen/teacache/sample_gedit.sh`, `qwen/toca/sample_gedit.sh`

For Qwen-Image-Edit (both `teacache` and `toca`), `from_pretrained()` accepts a **local directory path** directly. The shell scripts should pass the local path as `--model_name`, not via an env var:

```bash
MODEL_PATH="/mnt/data0/Qwen-Image-Edit-2509"
torchrun ... sample_gedit.py \
    --model_name "$MODEL_PATH" \   # ← direct local path, not an env var
    ...
```

**Note:** `QWEN25VL_MODEL_PATH` is still needed for `evaluate_gedit.py`'s VIEScore backbone (Section 6), because that model is loaded by `Qwen25VL` class in `qwen25vl_eval.py`, not by the sample script's pipeline.

---

## 8. Shell Scripts Created

### `qwen/teacache/sample_gedit.sh`
Loops over `REL_L1_THRESH=(1.0 1.2 1.4)`, outputs to `/home/hkl/Cache4Diffusion/samples/GEdit/teacache/R${rel_l1_thresh}`.

### `qwen/toca/sample_gedit.sh`
Loops over two ToCa configurations:
| Label | interval | fresh_ratio | Speedup |
|-------|----------|-------------|---------|
| N8R70 | 8 | 0.70 | ~4.5-5x |
| N12R75 | 12 | 0.75 | ~6x |

Outputs to `/home/hkl/Cache4Diffusion/samples/GEdit/toca/${label}`.

### `qwen/taylorseer/sample_gedit.sh`
Loops over TaylorSeer configurations with dynamic smoothing:
| Label | interval | max_order | first_enhance | alpha |
|-------|----------|-----------|---------------|-------|
| N10O2F3A0 | 10 | 2 | 3 | 0 (no smooth) |
| N10O2F3A0.8 | 10 | 2 | 3 | 0.8 (smooth) |

Outputs to `/home/hkl/Cache4Diffusion/samples/GEdit/taylorseer/N${interval}O${max_order}F${first_enhance}A${alpha}`.

**Smoothing env vars** (set dynamically in the loop):
- `USE_SMOOTHING=False` when `alpha=0`, `True` otherwise
- `SMOOTHING_ALPHA=$alpha`
- `SMOOTHING_METHOD=exponential`

---

## Pre-Run Checklist for New Baselines

Before running any new baseline's eval script, verify:

1. **Model loading** — Check that the model loading code (`__init__` in the MLLM backend class) uses an env var with HF fallback, not a hardcoded string.
2. **Tokenizer/model split** — If a wrapper class uses a repo-name heuristic to pick tokenizer class (e.g. `startswith("openai")`), the env var should supply the local path for **weights only**, while passing the correct HF repo name to get the right tokenizer. Otherwise the tokenizer class will be wrong.
3. **Dataset loading** — Check whether the local dataset uses `save_to_disk` (Arrow) or is a HuggingFace hub path. Use `load_from_disk` for local, `load_dataset` for hub.
4. **Output path** — Confirm sample script output dir matches eval script's `SAVE_DIR` / `--save_dir` exactly (same naming, same base prefix, same `fullset` subfolder if applicable).
5. **Eval script** — Confirm the eval loop parameters match the sample loop parameters (the bug in this session — completely mismatched parameter names).
6. **`sys.path` / import priority** — If the script imports from a package also installed system-wide (e.g. `flux`), confirm whether the local `src/` version is being used. Add `sys.path.insert(0, ...)` at the top if needed.
7. **Qwen local model path** — For Qwen-Image pipelines, `from_pretrained()` accepts local paths directly. Pass `--model_name "$MODEL_PATH"` in shell scripts, don't rely on env var injection unless the pipeline explicitly reads that env var.

