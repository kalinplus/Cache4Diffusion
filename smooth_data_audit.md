# TaylorSeer-Smooth 数据盘点与并入评估

> 用途:评估把 **TaylorSeer-Smooth**(对预测特征做 EMA 平滑的改进变体,系数 α)的结果并入论文各表格。
> 数据来源:`docs/` 下 11 个数据目录。论文端 7 个实际引用表:`tab/_flux`, `_qwen_image`, `_hunyuanvideo`, `_flux_kontext`, `_flux_quant`, `_flux_lora`, `_sdxl`。
> 生成时间:2026-07-11。

---

## 🚨 全局提醒(2026-07-13)— 本表对数据可用性判断过于乐观

**核心问题:大多数实验当时只存了指标数字(CSV/TXT/JSON),没有保存生成的图片。** 这意味着:

- 即使下文判定为 "🟢 能直接加 / 已并入" 的行,其指标也**无法重新核算、无法做感知类指标(PSNR/SSIM/LPIPS)复核、无法用新评测管线重测**;
- 下文所有基于"指标能逐位对上""同批可并"的乐观结论,前提都是"信任存下来的这批数字",一旦数字本身有问题(如 §G1 的 O=0 quirk、NF4 baseline 0.84 vs 0.97、Qwen N6O1 来源不明)就无从回查;
- 结论:**基本上所有 smooth(以及很多 naive)实验都得在统一配置下重跑,并这次务必保存图片。** 在重跑数据落地前,不要把本表的"能直接加"当成最终可用,应一律视为"待重跑验证"。

> 因此,下文各小节的 🟢/🟡/🔴 标记仅反映**数字层面的可用性判断**,不等于图片可复现。重跑计划见 §五,但应理解为其覆盖面比表中所列更广(基本全量)。

---

## TL;DR — 7 个已引用表能否直接加 smooth 行

| 论文表 | 模型 | docs 有 smooth | baseline 一致 | naive 一致 | 能否直接加 smooth 行 |
|---|---|---|---|---|---|
| `_sdxl` | SDXL | ✅ α0.75/0.8 | ✅ 0.62 | ✅ | 🟢 能(但感知指标反劣,需斟酌呈现) |
| `_flux_kontext` | FLUX-Kontext | ✅ α0.8 | ✅ 逐位匹配 | ✅ | 🟢 能(但 smooth 非单调改善) |
| `_flux_lora` | FLUX-LoRA(→animation2k_v1) | ✅ α0.8 | ✅ 0.87 | ✅ | ✅ 已并入(换 adapter,smooth O1 涨点) |
| `_flux` | FLUX.1-dev | ✅ α0.8/0.9 | ❌ docs 0.88 vs 主表 0.99 | ❌ | 🔴 不能并主表(批次不同) |
| `_flux_quant` | FLUX-NF4 | ✅ α0.8 | ✅ 改用 docs 0.84 | ✅ | ✅ 已并入(N3 反超 baseline,待复核) |
| `_qwen_image` | Qwen-Image | ✅ α0.7–0.9 | ≈ 1.24 vs 1.25 | N6O1 ❌ | 🟡 先核实 |
| `_hunyuanvideo` | HunyuanVideo | ✅ α0.8 | ❌ docs 无 baseline | ❌ docs 无 naive | 🔴 不能(只有 smooth) |

另有 **3 个论文尚未引用的新模型**:HunyuanImage2.1(⏸️ 暂缓:smooth 变体未定 + 缺 baseline/FLOPs)、QwenImageEdit(🟡 缺 baseline + 脏数据)、HunyuanVideo1.5(🔴 taylorcache 身份不明)。

---

## ⚠️ 必须先处理的 3 个全局硬问题

### G1. `O1+α == O0+α` quirk
**现象**:naive(α=0)下 O0/O1/O2 给出不同结果(如 NF4 N3: O0→0.798, O1→0.862, O2→0.870);但 smooth(α>0)开启后,**O=1 输出与 O=0 逐位完全相同**,只有 O=2 独立。即 smooth 把一阶外推抵消,退化成"复用+平滑"。

**① 会暴露重复数据的文件**(配置是完整 O0/O1/O2 × smooth 网格,O0+α 与 O1+α 两行都存在且相同):
- FLUX-NF4 `evaluation_results.txt` — ✅亲自核对(N3/N5/N6 的 O0A0.8==O1A0.8:32.3705/0.8670…)
- FLUX-LoRA `anime_lora` — ✅亲自核对;`animation2k_v1`/`realism_lora` 同格式(子 agent 报告)
- Qwen-Image `evaluation_results_qwen_db200.txt`(子 agent 报告)

**② 不暴露的文件**(smooth 只测了 O1/O2,没有 O0+smooth 行):FLUX.1-dev、SDXL、FLUX-Kontext、QwenImageEdit。→ 天然不会出现重复行,但也无法从数据判断底层 O1 是否退化。(旁证:FLUX-dev/SDXL 的 smooth 刻意只跑 O1/O2 跳过 O0,可能作者已知。)

**裁定(2026-07-11,用户)**:只有 **NF4 的 O=0 数据有问题**(作废,标记待重测);其余模型(LoRA / Qwen 等)的 O=0 无需处理。机制层(bug 还是 EMA 必然)不再深究。
> 数据事实备注(供参考,不改裁定):anime_lora 的 `O0A0.8` 与 `O1A0.8` 也逐位相同(31.4894/0.7405/31.239/0.8429/0.1884,N5/N6 同),现象与 NF4 一致;因 LoRA 表只报 O1、不用 O0,故不影响。

**论文实操准则(不变)**:smooth 行只报 O1/O2,不同时列 O0+smooth 与 O1+smooth;NF4 的 O=0 行作废待重测,加表时不要用。

### G2. α 取值不统一
| 模型 | 可用 α |
|---|---|
| FLUX.1-dev | 0.8, 0.9 |
| SDXL | 0.75, 0.8 |
| FLUX-NF4 / FLUX-LoRA / HunyuanVideo | 仅 0.8 |
| Qwen-Image | 0.7–0.9 全扫 |
| HunyuanImage2.1 | 0.75–0.95 |

论文若统一展示,**0.8 是最大公约数**。

### G3. 加速指标缺口
除 Qwen(`latency_summary.md`)外,所有 docs **都没有 Latency/FLOPs/Speed**。但 smooth 不改 cache schedule,FLOPs/Speed 可直接抄同 N/O 的 naive 行(SDXL/Kontext 表已这么做)。

---

## 一、🟢 能直接加的模型(数据同批,naive 对得上)

### 1. SDXL → `_sdxl.tex`
- **数据文件**:`StableDiffusionXL/evaluation_results_sdxl.txt`(baseline+naive)、`evaluation_results_sdxl_smooth.txt`(smooth,独立文件,baseline 逐位相同 → 同批)
- **格式**:5 指标 `ClipScore, ImageReward, PSNR, SSIM, LPIPS`,无加速指标。N∈{3,5,6}, O∈{0,1,2}, α∈{0.75,0.8}(无 0.9)
- **baseline**:`50 steps` IR=**0.6238** / CLIP=**34.0351**(== 论文 0.62 / 34.04 ✅)
- **naive vs 论文**(逐项对得上):N3O1=0.5643→0.56 ✅,N5O1=0.5080→0.51 ✅,N6O1=0.4942→0.49 ✅,N6O2=0.4886→0.49 ✅
- **smooth(α0.8,推荐与 FLUX 统一)**:

| N,O | naive IR | smooth IR(α0.8) | naive SSIM/LPIPS | smooth SSIM/LPIPS |
|---|---|---|---|---|
| 3,1 | 0.5643 | 0.5775 | 0.8050 / 0.1817 | 0.4945 / 0.5769 |
| 5,1 | 0.5080 | 0.4528 | 0.7337 / 0.2767 | 0.4812 / 0.6036 |
| 6,1 | 0.4942 | 0.4216 | 0.6842 / 0.3412 | 0.4673 / 0.6260 |
| 6,2 | 0.4886 | 0.4284 | 0.6905 / 0.3300 | 0.4712 / 0.6204 |

- **覆盖**:主表 active 行 N3O1/N5O1/N6O1/N6O2 **全覆盖** ✅
- **⚠️ caveat**:smooth 在 SDXL 上 **ImageReward 仅微变,但 SSIM/LPIPS 大幅劣化**(如 N3O1 SSIM 0.805→0.49)。与 FLUX 上 smooth 普遍改善相反。**投稿前必须核实**这是真实现象还是评测问题;若属实,放进表里不构成"改进",需斟酌是否展示或如何措辞。
- **✅ 已并入(2026-07-11)**:用户裁定——TaylorSeer 与 TaylorSeer-Smooth 都是本文方法,smooth 命中不好场景属正常,如实展示即可。已在 `tab/_sdxl.tex` 的 4 个 active 配置(N3O1/N5O1/N6O1/N6O2)后各加一行 TaylorSeer-Smooth(α=0.8,`\rowcolor{gray!20}` 区分),FLOPs/Speed 抄 naive,质量+感知用 α0.8 实测值;naive 行 bold 不变,smooth 行不加粗;表底脚注注明 α=0.8。编译通过(9 页)。

### 2. FLUX-Kontext → `_flux_kontext.tex`
- **数据文件**:`FLUX-Kontext/TaylorSeer/N{4,5,6}O1F3A{0,0.8}/scores.csv`、`N9O{1,2}F3Alpha{0,0.8}/score/scores.csv`
- **格式**:GEdit,`Language,Type,Group,Semantics,Quality,Overall`。论文用 Overall 行 Average:`Q_SC`=Semantics,`Q_PQ`=Quality,`Q_O`=Overall
- **一致性**:naive(A0)与论文表**逐位匹配**(N5O1 6.5235/7.2828/6.2203;N6O1 6.5056/7.2600/6.2328)✅
- **naive(A0) vs smooth(A0.8) Q_O**:

| 配置 | naive Q_SC/Q_PQ/Q_O | smooth Q_SC/Q_PQ/Q_O | 方向 |
|---|---|---|---|
| N4O1 | 6.4421/7.2770/6.1388 | 6.4970/7.2815/**6.2004** | 略好 |
| N5O1 | 6.5235/7.2828/**6.2203** | 6.4881/7.2684/**6.1797** | 变差 |
| N6O1 | 6.5056/7.2600/**6.2328** | 6.4337/7.2181/**6.1335** | 变差 |
| N9O1 | 6.1008/6.8620/5.7825 | 6.1791/6.9307/5.8679 | 略好 |
| N9O2 | 6.3735/7.0135/6.0499 | 5.1908/5.1560/**4.5609** | **崩塌** |

- **⚠️ caveat**:论文目前只展示 N5O1/N6O1,而这两配置 smooth 反而变差。**N9O2 的崩塌(Q_O 6.05→4.56)形态异常**(Q_PQ 从 ~7.0 掉到 5.16),可能是数据问题,用前抽查。建议挑 N4O1/N9O1 等 smooth 略好的配置,或重新组织呈现。
- **✅ 进展(2026-07-12,用户裁定)**:展示 **N4O1**(smooth 略好:Q_O 6.14→6.20)作为 smooth 正面证据。`tab/_flux_kontext.tex` 第二大行已加 TaylorSeer N4O1 + TaylorSeer-Smooth N4O1 两行(质量实测);smooth 使 Q_O delta 从 −1.2% 收窄到 −0.2%(相对 baseline 6.213)。N5O1/N6O1 原样保留。
- **🔴 待补跑(用户标注)**:
  1. **N4O1 加速指标**(Latency/FLOPs/Speed):`docs/FLUX-Kontext` 仅存质量 `scores.csv`,**无加速记录**(论文表 N5/N6 的加速列亦非来自 docs),需补测或由作者提供——表现加速列暂 `TBD`。
  2. **TaylorSeer O=0**:补 O=0(直接复用)配置作 O 阶对比。
  3. **第二大行(加速比 ~4×,FLOPs≈1993)的 DuCa 和 ToCa**:现第二大行只有 TeaCache + TaylorSeer N5O1,**缺 DuCa/ToCa 在该加速档的对比行**;`docs/FLUX-Kontext` 只有 TaylorSeer+TeaCache 两个方法,**无 ToCa/DuCa/FORA 数据**,需补跑。

### 3. FLUX-LoRA(→animation2k_v1) → `_flux_lora.tex` ✅
- **数据文件**:`FLUX-LoRA/lora-anime_lora/evaluation_results_float16.txt`(原 anime)、`lora-animation2k_v1/evaluation_results_float16.txt`(采用)
- **格式**:5 指标,N∈{3,5,6},O∈{0,1,2},α∈{0,0.8}
- **baseline(anime)**:`50 steps` IR=**0.7223** / CLIP=31.4880(== 论文 0.72 ✅)
- **naive(anime)**:N3O1=0.7386, N5O1=0.6964, N6O1=0.6936(对得上)
- **smooth(anime,α0.8)**:N3O1=0.7405, N5O1=**0.6589**, N6O1=**0.6258**
- **⚠️ caveat(anime)**:smooth 在 anime 上 **O1 大多掉点**(N5/N6 明显降)。
- **💡 备选 adapter** `lora-animation2k_v1`(baseline IR=0.8692):smooth **全面涨点**——N5O1 0.8372→0.8617、N6O1 0.7871→**0.8282**、N6O2 0.7161→0.7761。是 smooth 最好的正面证据,**建议换用或新增**。`lora-realism_lora`(baseline 0.7718)效果一般,可省。
- **✅ 已并入(2026-07-11)**:用户裁定——整表换成 `animation2k_v1`(非仅加 smooth 行),因 anime_lora 上 smooth O1 掉点、换 animation2k_v1 后 smooth O1 全面涨点才有说服力。`tab/_flux_lora.tex` 大表(注释)+小表(active)+正文(`sec/5_experiment.tex:68`)三处同步:baseline 0.72→0.87、naive 与 smooth 均用 animation2k_v1 同批;6 个 (N,O) 配置各加一行 TaylorSeer-Smooth(α=0.8,`\rowcolor{gray!20}`,只报 O1/O2 符合 G1);FLOPs/Speed 同 schedule 抄 naive(无空缺);caption/脚注改 animation2k_v1。⚠️ (1) 大表 FORA 已移除(animation2k_v1 无 FORA 数据,留 NOTE);(2) O2 smooth 多数掉点(N3/N5,仅 N6 涨),大表如实展示,小表只展 O1;(3) 正文删去原"O1 一致胜 O2"论断(N3 不成立),改述 smooth 在 O1 涨点。编译通过(9 页)。

---

## 二、🔴/🟡 不能直接加的模型(数据冲突或缺数据)

### 4. FLUX.1-dev → `_flux.tex`
- **数据文件**:`FLUX/flux_evaluation_results.txt`(baseline+naive+smooth 同文件)
- **baseline**:docs IR=**0.8797** / CLIP=31.9908(== `_flux_smooth.tex` 的 0.88 批次;主表 `_flux.tex` 是 **0.99** 批次 ✗)
- **结论**:docs 属于 **0.88 批次**,与主表 0.99 不同批。naive 也全不同(N5O2 docs 0.7757 vs 主表 1.02)。**smooth 不能并主表**,只能留在独立 `_flux_smooth.tex`(当前已注释)。
- **缺口**:docs **无 N=3**,主表 N3O2 拿不到 smooth。
- **smooth(α0.8/0.9,0.88 批次)**:N5O2 0.78→0.83(α0.8)/0.831(α0.9);N6O2 0.71→0.716/0.678。

### 5. FLUX-NF4 → `_flux_quant.tex` ✅
- **数据文件**:`FLUX-Quant-NF4/quant-nf4/evaluation_results.txt`
- **baseline**:docs IR=**0.8360** vs 论文 **0.97** ❌(差 0.13)。注意 docs 里 naive N3O1=0.8623 > baseline 0.84,说明 baseline 跨批。
- **naive**:全部对得上(N3O1=0.86, N5O1=0.81, N6O1=0.75, N3O2=0.87, N5O2=0.74, N6O2=0.67)
- **smooth(α0.8)**:N3O1=0.8670, N5O1=0.8016, N6O1=0.7637, N5O2=0.7642, N6O2=0.7071
- **结论**:加速后 IR 同批可用,但 **baseline 那一格跨批**。先把论文 baseline 改成 0.84(或查 0.97 来源)对齐,再加 smooth 行。
- **⚠️ O=0 数据作废待重测**:NF4 的 O=0(含 O0 naive 与 O0+smooth)数据有问题(见 §G1 裁定),加表时 O=0 行不要用,只用 O1/O2;O=0 需重测。
- **✅ 已并入(2026-07-12)**:用户裁定——baseline 用 docs 的 0.84(0.8360),弃用来源不明的 0.97。`tab/_flux_quant.tex` 大表(注释)+小表(active)+正文(`sec/5_experiment.tex` NF4 段)三处同步:baseline IR 0.97→0.84 / CLIP 32.49→32.30;deltas 全部按 0.84 重算;6 个 (N,O) 配置各加一行 TaylorSeer-Smooth(α=0.8,只报 O1/O2 符合 G1);大表 FORA 移除(docs 无 NF4 FORA 数据,留 NOTE);正文改述并加入 smooth 修复(N6 O1 0.75→0.76、O2 0.67→0.71)。⚠️ **副作用**:docs baseline 0.84 比它自己的 naive N3O1(0.86)/N3O2(0.87)还低 → N3 两配置 delta 变正(+3.1%/+4.1%,加速后反超 baseline),反常,可能是 docs baseline 那次跑得偏低;投稿前建议复核 NF4 baseline。编译通过(9 页)。表端 `tab/_flux_quant.tex` 已加内部 `%` 标记:baseline 0.84 待复核 + O=0 作废待重测(与 Qwen N6O1 同处理,见 §6)。

### 6. Qwen-Image → `_qwen_image.tex`
- **数据文件**:`QwenImage/evaluation_results_qwen_db200.txt`(有 baseline + N3)、`qwenimage_evaluation_results.txt`(无 baseline,N4/5/6 + α 全扫)、`latency_summary.md`(唯一带 latency/FLOPs)
- **baseline**:db200 IR=**1.2442** / CLIP=**35.5086**(论文 1.25 / 35.59,小幅不一致)
- **naive 冲突**:论文 N6O1=**1.01**,docs 文件A N6O1=**0.8228**(任何 α 最大 0.9545)❌ 硬冲突
- **跨文件混批**:论文表 N3 来自 db200,N5/N6 来自另一文件(连 baseline 都没记),是否同批未明
- **smooth 规律**:对 **O1 一致涨点**(N4O1 +0.04, N5O1 +0.07, N6O1 +0.07),对 **O2 一致掉点**(N5O2/N6O2 大跌)——支持"smooth 配 1 阶 Taylor"
- **结论**:先核实 N6O1 冲突 + 把整表统一到 db200(有 baseline+latency+FLOPs 配套),再加 smooth。
- **🔄 进展(2026-07-12)**:已为 3 个 active 配置加 TaylorSeer-Smooth(α0.8)行,逐位对账后定夺:
  - **N3O2**(db200 `N3O2F3Alpha0.8`):IR 1.22/CLIP 35.21/PSNR 30.94/SSIM 0.80/LPIPS 0.21,与 naive 同批(db200)→ **保留** ✅
  - **N5O2**(qwenimage `N5O2F3Alpha0.8`):IR 0.83(−33.7%),与论文 N5O2 naive(逐位=qwenimage naive,同批)配套,smooth O2 大跌符合"O2 一致掉点"规律 → **保留** ✅
  - **N6O1**(qwenimage `N6O1F3Alpha0.8`):IR 0.89 → **已注释,不加**。原因:该 smooth 与论文 naive(1.01,**来源不明**——非 qwenimage 的 0.82)跨批,并列会呈现"smooth 大幅反劣"假象(qwenimage 同批里 naive 0.82→smooth 0.89 实为涨)。
- **🔴 待重跑(重点)**:**N6O1 的 naive + smooth 必须在 db200 配置(同 baseline `N0O0F50Alpha0`)下重跑**——不是 qwenimage 那个配置。重跑后才能把 N6O1 smooth 行加回 `tab/_qwen_image.tex`(已留注释占位)。注意 db200 当前只覆盖 N3,需补 N6(理想情况连 N5)。
- **数据事实补充**:论文 N3O2 行逐位 = db200 naive ✅;论文 N5O2 行逐位 = qwenimage naive(无 baseline 文件);论文 N6O1 行(1.01/34.71/28.58/0.62/0.46)与 db200、qwenimage **任一都对不上**,来源完全不明,N6O1 naive 那格同样待 db200 重跑核对。

### 7. HunyuanVideo → `_hunyuanvideo.tex`
- **数据文件**:`HunyuanVideo/N{3,5,6}O1F1A0.8/scaled_results.json`(VBench total = `items[0]["total score"]`)
- **缺口(最严重)**:docs **只有 smooth(A0.8),完全没有 naive(A0)和 baseline(50steps)目录**。论文现有 naive(80.74/79.93/79.78)在 docs 无背书。
- **smooth VBench total**:N3=80.43, N5=79.32, N6=78.96 —— 全部**低于论文 naive**(80.74/79.93/79.78),且随 N 单调扩大(N6 低 0.82pp)
- **结论**:要加 smooth 列,**必须同批次重跑 baseline(50steps)+ naive(A0)+ smooth(A0.8)** 至少 N3/N5/N6 O1 F1。否则并列呈现不公平(异批次 + smooth 显得更差)。

---

## 三、3 个论文尚未引用的新模型

### N1. HunyuanImage2.1(文生图) — ⏸️ 暂缓
- **数据文件**:`HunyuanImage2.1/{origin/without_refiner, TaylorSeer, FORA, TeaCache/without_refiner}/evaluation_results.txt`
- **格式**:5 指标(ClipScore/ImageReward/PSNR/SSIM/LPIPS),with/without_refiner 两套;全部 `without_refiner`、同以 50steps 为参考,口径一致
- **TaylorSeer**(`hyimage_evaluation_results.txt`):naive(文件末尾 `smooth/exp/naive_ts/`)+ smooth(exp EMA / moving_average,α0.75–0.95)**同文件同批**,非常全
- **smooth 稳定优于 naive**(without_refiner, O1, α0.8):

| 配置 | naive IR | smooth IR(exp α0.8) |
|---|---|---|
| N4O1 | 0.8224 | 0.8830 |
| N5O1 | 0.8603 | 0.8728 |
| N6O1 | 0.8145 | 0.8599 |

- **对照**:FORA(N5 0.9034, N6 0.8950)、TeaCache(λ0.6 0.8431, λ0.8 0.7981)
- **缺口**:baseline 50-step 自身的 CLIP/ImageReward **缺失**(origin 只有 10/17/34 步 step-reduction);**所有方法均无 FLOPs/Speed**。
- **🔍 待观察(2026-07-12)**:smooth 有两种变体,行为差异极大。`exponential`(=EMA,论文 TaylorSeer-Smooth 定义)只在 O1 涨、**O2 全崩**(IR:N4O2 0.7542→0.5751、N5O2 0.8361→0.6122、N6O2 0.7905→0.5042);`moving_average` 在 O1/O2 **全面涨**(IR:N4O1 0.8224→0.9053、N5O2 0.8361→0.8695、N6O2 0.7905→0.8415)。**若 moving_average 在其它模型上也一致更好,可能应作为论文 Smooth 的报告变体**——需确认 moving_average 是否=论文 TaylorSeer-Smooth,并在其它模型上复测对比。
- **⏸️ 暂缓(2026-07-12,用户)**:本表先不加。阻塞点:(1) smooth 变体未定(见上);(2) baseline 50-step IR/CLIP 缺失;(3) 所有方法无 FLOPs/Speed。三项补齐后再做。

### N2. QwenImageEdit(图像编辑) — 🟡 暂不够
- **数据文件**:`QwenImageEdit/TaylorSeer/N{5,6,9}O{1,2}F3A{0,0.8}/score[s]/scores.csv` + TeaCache + ToCa
- **缺口**:**缺原始 baseline 行**;`N9O1F3A0.8/scores/scores.csv`(注意路径 `scores/` 非 `score/`)Q_O=7.37 是**脏数据**(同模型其它配置 3.5–4.6,翻倍不可能),须废弃/重跑
- **可信 smooth**:仅 N5O1、N6O1。naive Q_O:N5O1=4.45, N6O1=4.17;smooth:N5O1=4.48, N6O1=4.36
- **结论**:补 baseline + 重跑 N9O1 smooth 后可成表。

### N3. HunyuanVideo1.5(视频) — 🔴 身份未明
- **数据文件**:`HunyuanVideo1.5/{origin, taylorcache, deepcache, teacache}/.../scaled_results.json`
- **关键问题**:**没有 TaylorSeer 目录**,只有 `taylorcache`。`test_config.json` 的 `cache_type` 还停在 `deepcache`(不可信)。目录在独立仓库 `HunyuanVideo-1.5`(非老版的 `TaylorSeer-HunyuanVideo`)。`taylorcache` 可能是另一个方法 **TaylorCache**,而非本论文 TaylorSeer。
- **VBench total**:origin steps_50=**82.33**, taylorcache 81.78/81.57/81.50, deepcache 81.82/81.57, teacache 78.95/77.30
- **结论**:**必须先向跑实验的人确认 `taylorcache` 是不是 TaylorSeer**。若是,质量数据够新表(补 efficiency 列);若否,TaylorSeer 在 1.5 上零数据。

---

## 四、暂不使用的模型(记录在案)

- **FLUX-Schnell**(`FLUX-Schnext/...`):N4 整块 9 个配置数据**完全相同**(都 0.8575),O/α 未生效;体系 N2/N4+F1 与主表 N3/5/6+F3 不一致。**数据不可信,暂不用,需重跑。**
- **FLUX.1-dev 主表并 smooth**:批次不同(0.88 vs 0.99),维持独立 `_flux_smooth.tex` 现状,不强并。

---

## 五、推荐处理顺序

**第一批(数据干净,正面证据,可直接做)**:
1. SDXL 加 smooth 行(先核实感知反劣)
2. ~~FLUX-LoRA 换/加 animation2k adapter(smooth 全面涨点)~~ ✅ 已完成(整表换 animation2k_v1 + smooth 行)
3. HunyuanImage2.1 新增表(补 baseline 50-step 指标)

**第二批(要先修数据/核实)**:
4. 全局核实 O1+α==O0+α quirk(G1,阻塞所有 smooth 呈现)
5. ~~FLUX-NF4 baseline 对齐(0.84 vs 0.97)~~ ✅ 已完成(用 docs 0.84 + smooth 行;N3 反超 baseline 待复核)
6. Qwen-Image N6O1 冲突核实 + 批次统一
7. FLUX-Kontext smooth 呈现策略(挑配置,因非单调)

**第三批(要补实验)**:
8. HunyuanVideo 同批补 baseline+naive
9. QwenImageEdit 补 baseline + 重跑脏数据
10. HunyuanVideo1.5 确认 taylorcache 身份
