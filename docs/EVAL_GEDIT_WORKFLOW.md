# eval_gedit.sh 创建工作流

为已有 `evaluate_gedit.py` 但缺少 `eval_gedit.sh` 的目录创建评测脚本。

## 输入

调用方需提供：
1. **代码父目录** — 如 `qwen/teacache/`
2. **参考 `eval_gedit.sh`** — 如 `flux/taylorseer/eval_gedit.sh`

## 步骤

### Step 1: 修复 `evaluate_gedit.py` (对照 BENCHMARK_FIXES.md)

检查 `BENCHMARK_FIXES.md` 中 Section 2, 3, 6 记录的修复是否已应用到目标目录：

| 检查项 | 文件 | 正确做法 |
|--------|------|---------|
| Qwen25VL 模型路径 | `viescore/mllm_tools/qwen25vl_eval.py` | `os.environ.get("QWEN25VL_MODEL_PATH", "Qwen/Qwen2.5-VL-72B-Instruct-AWQ")` |
| 数据集加载 | `evaluate_gedit.py` | `os.environ.get("GEDIT_DATASET_PATH", ...)` + `load_from_disk` + fallback |
| import | `evaluate_gedit.py` | `from datasets import load_dataset, load_from_disk` |

使用 `diff` 对比已修复的参考文件（如 `flux/taylorseer/evaluate_gedit.py`）确认一致。

### Step 2: 适配 `eval_gedit.sh`

从参考脚本改写，需适配以下字段：

| 字段 | 来源 |
|------|------|
| `cd` 目录 | 代码父目录 |
| `python` 调用路径 | `{PROJECT_ROOT}/{parent_dir}/evaluate_gedit.py` |
| `SAVE_DIR` 前缀 | `samples/GEdit/{model}/{method}/` |
| 配置循环变量 + 命名格式 | 该目录下的 `sample_gedit.sh` |
| 环境变量 (`GEDIT_DATASET_PATH`, `QWEN25VL_MODEL_PATH`) | 与参考脚本一致 |
| GPU 分配 | 调用方指定 |

额外逻辑：
- **跳过不完整配置** — fullset 为空或图片数量不足的配置（如 OOM 中断的）
- **fullset 存在性检查** — 跳过无 `fullset` 目录的配置

## 当前待处理

| 目录 | 状态 |
|------|------|
| `qwen/taylorseer/` | ✅ 已完成 |
| `qwen/teacache/` | ✅ 已完成 |
| `qwen/toca/` | ✅ 已完成 |
