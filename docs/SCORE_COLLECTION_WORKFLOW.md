# Score Collection Workflow

将测评结果从生成目录自动整理到统一的目标目录。

## 用法

告诉 Claude：

```
整理 scores，源目录 samples/GEdit/{model}/{method}，目标目录 samples/scores/{model}/{method}，只取 score/scores.csv
```

Claude 会：

1. 遍历源目录下的所有子目录
2. 列出每个子目录中 `score/scores.csv` 的存在情况
3. 展示整理计划表（源 → 目标）供确认
4. 确认后执行 `mkdir -p` + `cp`

## 目录约定

```
samples/GEdit/{model}/{method}/{config}/score/scores.csv   # 源
    ↓
samples/scores/{model}/{method}/{config}/score/scores.csv   # 目标
```

- `{model}` — 模型名，如 `flux-kontext`、`qwen2.5`（未来扩展）
- `{method}` — 方法名，如 `taylorseer`、`teacache`、`naive`（未来扩展）
- `{config}` — 超参配置，如 `N9O1F3Alpha0`、`R0.8`（由子目录名决定）

只保留 `scores.csv`（汇总分数），不复制各子类分数 CSV。

## 示例

源：`samples/GEdit/flux-kontext/taylorseer/`
目标：`samples/scores/flux-kontext/taylorseer/`

| 源文件 | 目标 |
|--------|------|
| `.../taylorseer/N9O1F3Alpha0/score/scores.csv` | `.../scores/.../taylorseer/N9O1F3Alpha0/score/scores.csv` |
| `.../taylorseer/N9O1F3Alpha0.8/score/scores.csv` | `.../scores/.../taylorseer/N9O1F3Alpha0.8/score/scores.csv` |
| ... | ... |
