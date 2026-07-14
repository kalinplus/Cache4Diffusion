#!/usr/bin/env bash
# =============================================================================
# FLUX.1-schnell TaylorSeer 配置扫描脚本 (通过 run.py 调度)
# FLUX.1-schnell TaylorSeer config sweep, dispatched through run.py.
#
# 跑哪些配置 / What it runs (per config, 2 run.py calls):
#   • [bench] 单 prompt 测速: latency + transformer FLOPs → <config>/benchmark.txt
#   • [gen]   DrawBench200 (200 条 prompt) 生图; 缓存配置额外跑 --eval
#
# 配置清单 / Configs:
#   1. origin / baseline (无缓存, --no_cache → run.py 注入 interval=1)
#   2. 缓存扫描 (笛卡尔积) / cached sweep:
#        N(cache_interval) ∈ {1,2} × O(cache_max_order) ∈ {0,1}
#        × F(cache_first_enhance)=1 × Alpha(smoothing) ∈ {0, 0.8}
#
# 为什么每个配置要跑两次 run.py? / Why two calls per config?
#   run.py 规定 --benchmark 与 --eval 互斥, 且 --benchmark 强制单 prompt.
#   所以测速 (单 prompt) 与 200 条生图+评测必须分开两次调用, 各自加载一次模型。
#   run.py forces --benchmark ⊥ --eval and --benchmark uses a single prompt, so
#   speed (1 prompt) and 200-image gen+eval are separate calls (2 model loads/config).
#
# 评测说明 / Eval note:
#   baseline 自身不跑 eval (它是参考集, 自评会得到 PSNR=inf/SSIM=1/LPIPS=0, 无意义)。
#   8 个缓存配置各自 --eval, 自动以 sibling baseline/S4 为参考图, 得到
#   CLIP/ImageReward (绝对质量) + PSNR/SSIM/LPIPS (相对 baseline 的保真度)。
#
# 输出布局 / Output layout:
#   outputs/flux_schnell/baseline/S4/{img_0..199.jpg, benchmark.txt}
#   outputs/flux_schnell/taylorseer/S4_N{..}O{..}F{..}[A0.8]/
#       {img_0..199.jpg, benchmark.txt, evaluation_results.txt}
#
# 用法 / Usage:
#   bash scripts/flux_schnell_sweep.sh                  # 默认: DrawBench200, GPU 0, eval+bench 开
#   GPU=5 bash scripts/flux_schnell_sweep.sh            # 换卡
#   BENCH=0 bash scripts/flux_schnell_sweep.sh          # 只生图+评测, 不测速
#   EVAL=0  bash scripts/flux_schnell_sweep.sh          # 只生图+测速, 不评测
#   DRY_RUN=1 bash scripts/flux_schnell_sweep.sh        # 预览全部 18 条命令, 不执行
# =============================================================================
set -uo pipefail

cd "$(dirname "$0")/.."   # 仓库根目录 / repo root (contains run.py)

# 把 HF 缓存钉在本机真实存在的位置。run.py 的 run_eval 默认把 XDG_CACHE_HOME 设成
# /data/public/.cache (本机不存在该路径), 会让 transformers 找不到已缓存的
# bert-base-uncased 进而尝试下载到一个不存在的 /data → eval 崩溃。导出有效路径即可覆盖。
# Pin the HF cache to a path that exists on this host; run.py's run_eval otherwise
# defaults XDG_CACHE_HOME to an absent /data/... and the ImageReward/BLIP bert
# download fails. Override-able via the environment.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"

# ── 可调参数 / Tunables ─────────────────────────────────────────────────────
GPU="${GPU:-0}"
MODEL="${MODEL:-flux_schnell}"
SEED="${SEED:-0}"
STEPS="${STEPS:-4}"                     # schnell 强制 4 步 (sample.py asserts ==4)
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"

# 生图 prompt 文件 (默认 DrawBench200, 200 条)。设 PROMPT 则改单条。
PROMPT_FILE="${PROMPT_FILE:-assets/prompts/DrawBench200.txt}"
PROMPT="${PROMPT:-}"                    # 非空则覆盖 PROMPT_FILE, 跑单条

# 扫描旋钮 / sweep knobs (笛卡尔积 / Cartesian product).
INTERVALS=(1 2)        # N = cache_interval
MAX_ORDERS=(0 1)       # O = cache_max_order
FIRST_ENHANCE=(1)      # F = cache_first_enhance
ALPHAS=(0 0.8)         # Alpha = smoothing_alpha (0 → smoothing off; >0 → on)

EVAL="${EVAL:-1}"      # 1 → 缓存配置生成后自动评测 (需要 eval conda 环境)
BENCH="${BENCH:-1}"    # 1 → 每个配置单 prompt 测速 (latency + FLOPs)
BENCH_PROMPT="${BENCH_PROMPT:-a red panda wearing a top hat, photorealistic, highly detailed}"
BENCH_WARMUP="${BENCH_WARMUP:-1}"
BENCH_RUNS="${BENCH_RUNS:-3}"           # 计时次数, 取均值; schnell 很快, 3 次足够稳

DRY_RUN="${DRY_RUN:-0}"                 # 1 → 只打印每个 run.py 计划, 不执行 (预览 18 条命令)

# ── 公共参数 (不含 prompt/eval/benchmark) / common flags ────────────────────
COMMON=(--model "$MODEL" --gpu "$GPU" --seed "$SEED" --steps "$STEPS"
        --width "$WIDTH" --height "$HEIGHT")

# 生图参数: prompt_file 或单 prompt / gen input flags.
if [ -n "$PROMPT" ]; then
  GEN_INPUT=(--prompt "$PROMPT")
else
  GEN_INPUT=(--prompt_file "$PROMPT_FILE")
fi

# ── 计数 / tallies ──────────────────────────────────────────────────────────
FAIL=0
OK=0
_call() {  # $1=tag  rest=run.py argv
  local tag="$1"; shift
  echo "  [$tag] python run.py $*"
  local rc=0
  if [ "${DRY_RUN:-0}" = 1 ]; then
    python run.py --dry_run "$@" || rc=$?   # 预览: 只打印计划, 不执行 / preview only
  else
    python run.py "$@" || rc=$?
  fi
  if [ "$rc" -eq 0 ]; then
    OK=$((OK+1))
  else
    echo "    ✗ FAILED (rc=$rc)"
    FAIL=$((FAIL+1))
  fi
}

# ── 单个配置: 测速 + 生图(+评测) / one config: bench + gen(+eval) ───────────
#   $1 label   $2 do_eval(1/0)   rest = run.py cache flags (--no_cache or --cache_*/--use_smoothing ...)
run_config() {
  local label="$1"; local do_eval="$2"; shift 2
  local cfg=("$@")

  echo ""
  echo "=================================================================="
  echo "▶ $label"
  echo "=================================================================="

  # [bench] 单 prompt 测速 (与 --eval 互斥, 单独一次调用)。
  if [ "$BENCH" = 1 ]; then
    _call bench "${COMMON[@]}" --prompt "$BENCH_PROMPT" --benchmark \
        --benchmark_warmup "$BENCH_WARMUP" --benchmark_runs "$BENCH_RUNS" "${cfg[@]}"
  fi

  # [gen] 200 条生图 (+ eval for cached configs)。
  local gen=("${COMMON[@]}" "${GEN_INPUT[@]}")
  [ "$do_eval" = 1 ] && gen+=(--eval)
  _call gen "${gen[@]}" "${cfg[@]}"
}

# ── 1) origin / baseline (无缓存; 不自评, 作为参考集) ───────────────────────
run_config "origin (baseline, no cache)" 0 --no_cache

# ── 2) cached sweep: N × O × F × Alpha ──────────────────────────────────────
for n in "${INTERVALS[@]}"; do
  for o in "${MAX_ORDERS[@]}"; do
    for f in "${FIRST_ENHANCE[@]}"; do
      for a in "${ALPHAS[@]}"; do
        if [ "$a" = 0 ]; then
          run_config "cached  N${n} O${o} F${f} (no smoothing)" "$EVAL" \
            --cache_interval "$n" --cache_max_order "$o" --cache_first_enhance "$f"
        else
          run_config "cached  N${n} O${o} F${f} Alpha=${a}" "$EVAL" \
            --cache_interval "$n" --cache_max_order "$o" --cache_first_enhance "$f" \
            --use_smoothing --smoothing_alpha "$a" --smoothing_method exponential
        fi
      done
    done
  done
done

# ── 汇总 / Summary ─────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "✓ sweep finished: $OK ok, $FAIL failed."
echo "  outputs under: outputs/flux_schnell/{baseline,taylorseer}/"
echo "  per cached config: benchmark.txt + evaluation_results.txt + img_*.jpg"
echo "=================================================================="
[ "$FAIL" -eq 0 ]
