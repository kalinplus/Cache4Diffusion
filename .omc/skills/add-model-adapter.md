---
id: add-model-adapter
name: Add Model Adapter to Cache4Diffusion
description: Step-by-step process for integrating a new diffusion model into the Cache4Diffusion Strategy+Adapter framework
source: learned
triggers:
  - "add model adapter"
  - "integrate new model"
  - "new adapter"
  - "model integration cache4diffusion"
quality: high
---

# Add Model Adapter to Cache4Diffusion

## The Insight

Cache4Diffusion decouples caching strategies (TaylorSeer, etc.) from model architectures via the Strategy+Adapter pattern. Adding a new model means implementing a `ModelAdapter` subclass that translates the generic cache interface into model-specific forward logic — without touching the strategy code at all.

## Why This Matters

Each diffusion model has unique preprocessing (embeddings, RoPE, attention masks), unique block structures (double-stream only, or double+single), and unique output reshaping (especially video models). The adapter isolates all of this so the caching math stays model-agnostic.

## Recognition Pattern

Use this skill when:
- A new model directory exists under the project root (e.g., `hunyuan_video/`, `qwen_image/`)
- The model has a `forwards/` subdirectory with existing TaylorSeer forward overrides
- You need to wire it into `infer.py` / `batch_infer.py` via `--model_name`

## The Approach

### Step 0: Study the existing forward implementations

Read the model's existing forward files in `<model>/taylorseer_<model>/forwards/`:
- `<model>_forward.py` — top-level pipeline forward (preprocessing, block loop, output)
- `double_transformer_forward.py` — per-block full/cached logic for double-stream
- `single_transformer_forward.py` — per-block full/cached logic for single-stream (if exists)

Also read an existing adapter for reference: `model_adapters/adapters/flux_adapter.py` (has both streams) or `qwen_image_adapter.py` (double-stream only).

Key things to extract from the existing forward:
- What preprocessing does the model do? (embeddings, RoPE, attention mask construction)
- Does it have single-stream blocks? What do they return?
- What does the output look like? (video reshape? simple proj_out?)
- What kwargs does `block.attn()` expect? (`attention_mask`? `image_rotary_emb` vs `freqs_cis`?)

### Step 1: Create the adapter file

Create `model_adapters/adapters/<model_name>_adapter.py`.

**Skeleton:**
```python
from model_adapters.base import ModelAdapter
from model_adapters.info import ModelInfo
from caching_core import CacheStrategy

class <ModelName>Adapter(ModelAdapter):
    def get_model_info(self, model) -> ModelInfo: ...
    def get_block_iterators(self, model) -> Dict[str, List]: ...
    def create_forward_fn(self, model, strategy): ...
    def forward_double_block_full(...): ...
    def forward_double_block_cached(...): ...
    def forward_single_block_full(...): ...   # raise NotImplementedError if no single stream
    def forward_single_block_cached(...): ...
```

### Step 2: Implement `get_model_info`

```python
return ModelInfo(
    num_double_layers=model.config.num_layers,
    num_single_layers=model.config.num_single_layers,  # 0 if none
    has_double_stream=True,
    has_single_stream=bool(model.config.num_single_layers),
)
```

### Step 3: Implement `create_forward_fn`

This is the most model-specific part. Override the base class version entirely.

**Template:**
```python
def create_forward_fn(self, model, strategy):
    model_info = self.get_model_info(model)
    adapter = self

    def patched_forward(...model-specific signature...):
        # 1. Init cache on first call
        if attention_kwargs.get('cache_dic') is None:
            cache_dic = cache_init(model_info.num_double_layers, model_info.num_single_layers)
            ctx = StepContext(num_steps=model.num_steps)
            attention_kwargs['cache_dic'] = cache_dic
            attention_kwargs['current'] = ctx

        cache_dic = attention_kwargs['cache_dic']
        ctx = attention_kwargs['current']
        strategy.schedule_step(cache_dic, ctx)

        # 2. LoRA scale (copy attention_kwargs before popping 'scale')
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop('scale', 1.0)

        # 3. Model-specific preprocessing (embeddings, RoPE, attention mask)
        ...

        # 4. Double stream block loop
        ctx.stream = 'double_stream'
        for idx, block in enumerate(model.transformer_blocks):
            ctx.layer = idx
            if ctx.type == 'full':
                hidden_states, encoder_hidden_states = adapter.forward_double_block_full(...)
            else:
                hidden_states, encoder_hidden_states = adapter.forward_double_block_cached(...)
            strategy.on_block_end(cache_dic, ctx, hidden_states)

        # 5. Single stream block loop (if applicable)
        # NOTE: HunyuanVideo single blocks return (hidden_states, encoder_hidden_states)
        # FLUX single blocks return only hidden_states (after cat/split is done externally)

        # 6. Output projection + model-specific reshape
        ctx.step += 1
        return Transformer2DModelOutput(sample=output)

    return patched_forward
```

**Critical pitfalls:**
- Always `.copy()` `attention_kwargs` before `.pop('scale', ...)` — the dict is shared across steps
- `StepContext(num_steps=model.num_steps)` — `num_steps` is set on the class by `setup_pipeline`
- For video models: `time_text_embed` returns `(temb, token_replace_emb)` — unpack with `_`
- HunyuanVideo uses `attention_kwargs` (not `joint_attention_kwargs`) as the kwarg name

### Step 4: Implement block methods

**`forward_double_block_full`** — run the block, call `strategy.on_full_compute()` after each module output:
```python
ctx.module = 'img_attn'; strategy.on_full_compute(cache_dic, ctx, attn_output)
ctx.module = 'img_mlp';  strategy.on_full_compute(cache_dic, ctx, ff_output)
ctx.module = 'txt_attn'; strategy.on_full_compute(cache_dic, ctx, context_attn_output)
ctx.module = 'txt_mlp';  strategy.on_full_compute(cache_dic, ctx, context_ff_output)
```

**`forward_double_block_cached`** — skip computation, call `strategy.on_cache_restore()`:
```python
ctx.module = 'img_attn'
hidden_states = hidden_states + gate_msa.unsqueeze(1) * strategy.on_cache_restore(cache_dic, ctx)
# ... repeat for img_mlp, txt_attn, txt_mlp
```
Still run `block.norm1()` / `block.norm1_context()` to get the gate values — they're cheap and needed for residual scaling.

**Single-stream blocks (HunyuanVideo pattern):**
- Concatenate `hidden_states` and `encoder_hidden_states` before processing
- Cache the full `proj_output` (pre-residual) under `ctx.module = 'total'`
- Split back after residual add: `return hs[:, :-txt_len], hs[:, -txt_len:]`
- Pass `encoder_hidden_states` via `**kwargs` since base class signature only has `hidden_states`

**FLUX single-stream pattern (different):**
- FLUX concatenates txt+img externally before the single block loop
- Single blocks only see the concatenated tensor, return only `hidden_states`

### Step 5: Register in factory and scripts

**`model_adapters/adapters/__init__.py`:**
```python
from .hunyuan_video_adapter import HunyuanVideoAdapter
__all__ = [..., 'HunyuanVideoAdapter']
```

**`model_adapters/factory.py`** — add to `create_caching_pipeline`:
```python
elif model_name == 'hunyuan_video':
    from model_adapters.adapters import HunyuanVideoAdapter
    adapter = HunyuanVideoAdapter()
```

**`infer.py`** — add to `--model_name` choices and `_GUIDANCE_KWARG` / `_USE_DEVICE_MAP` dicts as needed.

**`batch_infer.py`** — same pattern as `infer.py`.

**`infer.sh`** — add a new `elif` branch with `model_path` and `dtype`, update the usage comment and error message:
```bash
elif [ "$MODEL_NAME" = "hunyuan_video" ]; then
    model_path=""  # TODO: fill in model path
    dtype="bfloat16"
```

**`test.sh`** — same change as `infer.sh` (test.sh mirrors infer.sh with `--steps 1`).

### Step 6: Update CLAUDE.md

Add the new model to the Migration Stages table and note any pending test status.

## Known Gotchas

- `block.norm1()` returns 5 values for double blocks: `(norm_hs, gate_msa, shift_mlp, scale_mlp, gate_mlp)` — in cached path you only need gates, use `_` for the rest
- `gate_msa` shape is `[B, D]` — always `.unsqueeze(1)` before multiplying with `[B, T, D]` tensors
- HunyuanVideo's `block.attn()` takes `attention_mask` as a positional-style kwarg; FLUX's does not
- `del rotary_emb` / `del kwargs` in cached methods to silence Pylance "not accessed" hints
- `patch_model_with_cache` in `factory.py` should unconditionally patch `model.forward` — no per-model branching needed
