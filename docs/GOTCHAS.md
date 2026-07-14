# Gotchas & bugfixes (reusable notes)

Pitfalls hit while wiring **FLUX.1-schnell** into `run.py` together with
`--benchmark` + `--eval`, with root cause, fix, and **when this recurs** so the
same patterns apply to the next model / variant.

Related docs:
- benchmark harness (latency + FLOPs) → [`BENCHMARK_FIXES.md`](BENCHMARK_FIXES.md),
  and `CLAUDE.md` → "Speed benchmark — latency + FLOPs"
- FLUX-schnell specifics (4-step assert, T5 maxlen 256, linear schedule,
  `guidance_embed=False`) → [`FLUX_SCHNELL_TAYLORSEER.md`](FLUX_SCHNELL_TAYLORSEER.md)
- the sweep script that drove all of this → `scripts/flux_schnell_sweep.sh`

---

## 1. Raw-FLUX `--no_cache` is NOT a true baseline (caching is purely parametric)

**Symptom.** For the raw-BFL FLUX runners (`flux`, `flux_schnell`,
`flux_kontext*`), `run.py ... --no_cache` produced a **cached** run (interval=4),
not a baseline. So the "origin" reference images were wrong.

**Root cause.** `flux/taylorseer/src/sample.py` **always** calls
`denoise_cache(interval, max_order, first_enhance)` — there is **no caching on/off
flag**. Caching is controlled purely by those three parameters. `run.py`'s generic
`--no_cache` only *skips* the cache knobs, so `sample.py` fell back to its argparse
defaults (`--interval 4`) → still accelerated.

A true baseline for this scheme is `interval=1` (every step is a *fresh* / full
step). Confirmed in `flux/.../cache_functions/cal_type.py`:
```python
first_step = (current['step'] < cache_dic['first_enhance'])
if (first_step) or (current['cache_counter'] == cache_dic['interval'] - 1):
    ...  # this step is a FRESH (full-compute) step
```
With `interval=1`, `cache_counter == interval-1 == 0` on every step → every step
fresh → no caching.

**Fix.** Added a `ModelRunner.no_cache_baseline` field to `run.py` (default
empty). Runners whose caching is parametric declare the cache values to *inject*
(rather than skip) under `--no_cache`. `flux_schnell` sets
`no_cache_baseline={"cache_interval": 1}`. Honored in `build_argv()` / `build_env()`.
Existing runners keep `{}` → unchanged behavior.

**Recurs when.** Wiring any model whose entry script has caching always-on via
parameters (the whole raw-BFL FLUX family; possibly other repos). Set
`no_cache_baseline` to whatever makes every denoise step a full step (usually
`{"cache_interval": 1}`). Diffusers-style pipelines that have a real on/off flag
(`--use_taylor`) do **not** need this.

---

## 2. `run.py --eval` cache path points at a non-existent `/data`

**Symptom.** `run.py ... --eval` crashed during scoring:
```
OSError: PermissionError at /data when downloading bert-base-uncased.
Check cache directory permissions.
```
(the BLIP tokenizer inside ImageReward).

**Root cause.** `run.py` `run_eval()` does
`run_env.setdefault("XDG_CACHE_HOME", "/data/public/.cache")`. On **this host**
`/data` does not exist at all (models live under `/mnt/workspace/hkl` and
`/mnt/data0`; the HF cache is at `$HOME/.cache/huggingface` with
`HOME=/mnt/workspace/hkl`). So transformers' cache got redirected to a path that
doesn't exist, it missed the **already-cached** `bert-base-uncased`, and tried to
download to `/data` → permission/path error.

`bert-base-uncased` *is* fully present at
`~/.cache/huggingface/hub/models--bert-base-uncased/` — the only problem was the
redirected cache root.

**Fix.** `scripts/flux_schnell_sweep.sh` exports the cache roots to a real path
before calling `run.py`:
```bash
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
```
Because `run_eval()` uses `setdefault`, exporting in the parent environment wins.

**Recurs when.** Running `run.py --eval` for **any** model on this host from a
script/shell that doesn't pin the cache. Either export the two vars above, or fix
`run.py`'s `/data/public/.cache` default to something `$HOME`-based. This is not
schnell-specific — it affects every model's `--eval` here.

---

## 3. `sample.py --benchmark`: `UnboundLocalError` on `prompts` / `base_seed`

**Symptom.** `run.py --model flux_schnell --benchmark ...` crashed:
```
UnboundLocalError: local variable 'prompts' referenced before assignment
# after that fix:
UnboundLocalError: local variable 'base_seed' referenced before assignment
```
The `--benchmark` **dry-run** looked fine (it only builds the command); the real
run hit this.

**Root cause.** In `flux/taylorseer/src/sample.py` `main()`, the optional
benchmark early-return block references `prompts[0]` and `base_seed`, but **both
are assigned later in the same function** (`prompts = opts.prompts` and
`base_seed = opts.seed / torch.randint(...)` sit *after* the block). Python
therefore treats them as local for the whole function → unbound at the reference
inside the block. The normal (non-benchmark) path worked because it skips the
block and assigns before its own first use.

**Fix.**
- `bench_prompt = prompts[0]` → `bench_prompt = opts.prompts[0]`
  (`opts.prompts` is populated from `SamplingOptions`).
- Move the `base_seed` computation to **before** the benchmark block (both the
  benchmark path and the normal loop need it).

**Recurs when.** Adding a `--benchmark` early-return block to **any** model entry
script: every variable the block touches must be bound **before** the block (or
read off `opts.*`). This was latent in the shared flux `sample.py`, so it silently
broke `flux` (dev) `--benchmark` too — now fixed for both. Audit new entry
scripts for forward references before the benchmark `return`.

---

## 4. (Context, not a bug) `flux_schnell` had no `run.py` runner

`run.py` shipped only FLUX.1-dev runners (`flux`, `flux_diffusers`), both
hardcoded to dev (`model_name="flux-dev"`, `flux1-dev.safetensors`). Added a
`flux_schnell` `ModelRunner` cloned from `flux` with:
- `model_name="flux-schnell"`, `env_builder` → `flux1-schnell.safetensors`
- `defaults`: `steps=4` (sample.py asserts 4 for schnell), no `guidance`
  (schnell's `guidance_embed=False` ignores it)
- smoothing env map (`USE_SMOOTHING` / `SMOOTHING_ALPHA` / `SMOOTHING_METHOD`)
  — the dev `flux` runner didn't forward these
- `no_cache_baseline={"cache_interval": 1}` (see §1)
- `DEFAULT_MODEL_PATHS["flux_schnell"]` → `.../FLUX.1-schnell`

**Recurs when.** Adding any new model variant: clone the closest runner, then
adjust `model_name` / weights filename / `steps` / `dtype` / cache defaults, and
decide whether it needs `no_cache_baseline` (parametric caching) or a `cache_flag`
(on/off flag).

---

## Quick checklist — wiring a new model with `--benchmark` + `--eval`

1. **Baseline semantics.** Does the entry script have a real caching on/off flag?
   - Yes (diffusers `--use_taylor`, …) → set `cache_flag`.
   - No (raw BFL FLUX, always-via-params) → set `no_cache_baseline={"cache_interval": 1}`.
2. **Benchmark block.** If you add/edit a `--benchmark` early-return block, every
   var it uses must be bound *before* the block (or read from `opts.*`). No
   forward references.
3. **Eval cache.** On this host, export `HF_HOME` / `XDG_CACHE_HOME` to
   `$HOME/.cache/...` before `run.py --eval`, or the ImageReward/BLIP bert
   download fails at `/data`.
4. **Weights & T5/CLIP.** Raw-FLUX models take weights via env
   (`FLUX_MODEL`/`FLUX_AE`/`T5_MODEL_PATH`/`CLIP_MODEL_PATH`); the `env_builder`
   picks the right `flux1-*.safetensors` per variant.
5. **Schnell constraints.** `num_steps` must be 4; guidance is ignored; T5
   `max_length=256`; linear schedule (no shift) — all handled inside `sample.py`
   keyed on `model_name == "flux-schnell"`.
