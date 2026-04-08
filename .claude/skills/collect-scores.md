Collect and organize evaluation score files into a unified target directory.

## Trigger Keywords
"collect scores", "collect-scores", "整理分数", "整理测评", "整理指标", "汇总分数", "汇总指标", "汇总测评"

## Arguments
- `$ARGUMENTS`: Description of source directory, target directory, and optionally which files to collect (default: `score/scores.csv`)

## Workflow

1. **Parse input**: Extract source and target directory from `$ARGUMENTS`. Default file pattern is `score/scores.csv`. If user specifies other files, use those instead.

2. **Scan source**: List all immediate subdirectories under the source directory. For each subdirectory, check if the target file (`score/scores.csv` by default) exists.

3. **Present plan**: Show a table with all files to be copied (source -> target). Ask user for confirmation before proceeding.

4. **Execute**:
   - `mkdir -p` for each target directory
   - `cp` each file
   - `ls -l` to verify all copies succeeded

5. **Report**: Brief summary of how many files were copied.

## Directory Convention

```
源: samples/GEdit/{model}/{method}/{config}/score/scores.csv
目标: samples/scores/{model}/{method}/{config}/score/scores.csv
```

The directory structure under `{model}/{method}/` is preserved exactly — subdirectory names are not modified.

## Rules

- NEVER copy files without user confirmation first
- NEVER delete or modify source files
- If a target file already exists, warn the user and ask whether to overwrite
- Only copy the specified file(s), not the entire score directory
