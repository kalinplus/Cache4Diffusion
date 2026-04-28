# 各个模型对应的目录

- FLUX.1-dev (原版):
    - 原版：/home/hkl/Cache4Diffusion/flux
    - Diffusers 版：/home/hkl/Cache4Diffusion/flux_simple

- FLUX-Schnell
    - /home/hkl/Cache4Diffusion/flux

- HunyuanVideo:
    - /home/hkl/Cache4Diffusion/hunyuan_video

- HunyuanVideo-1.5
    - /home/hkl/Cache4Diffusion/HunyuanVideo-1.5

- Qwen-Image
    - /home/hkl/Cache4Diffusion/qwen

- Qwen-Image-Edit
    - /home/hkl/Cache4Diffusion/qwen (不同参数设置)

- HunyuanImage-2.1
    - /home/hkl/Cache4Diffusion/HunyuanImage-2.1

- FLUX-Quant-NF4
    - /home/hkl/Cache4Diffusion/flux_simple

- FLUX-LoRA
    - /home/hkl/Cache4Diffusion/flux_simple

- FLUX-Kontext
    - /home/hkl/Cache4Diffusion/flux_simple

- Stable-Diffusion XL
    - /home/hkl/Cache4Diffusion/sdxl

<!-- - DiT
    - /home/hkl/Cache4Diffusion/dit

- HiDream
    - /home/hkl/Cache4Diffusion/taylorseer_upstream/TaylorSeer-HiDream

- Wan2.1
    - /home/hkl/Cache4Diffusion/taylorseer_upstream/TaylorSeer-Wan2.1

- FramePack
    - /home/hkl/Cache4Diffusion/taylorseer_upstream/TaylorSeer-FramePack

- FLUX-Kontext (upstream)
    - /home/hkl/Cache4Diffusion/taylorseer_upstream/TaylorSeer-FLUX-Kontext -->


## 各个模型适配的方法情况

| 方法 | FLUX<br>(原版) | FLUX<br>(Diffusers) | Qwen-Image | HunyuanVideo | HunyuanVideo-1.5 | HunyuanImage-2.1 | SDXL |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| TaylorSeer | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| FoRA | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| ToCa | ✅ | - | ✅ | - | - | - | - |
| DuCa | ✅ | - | ✅ | - | - | - | - |
| FreqCA | ✅ | - | ✅ | - | - | - | ✅ |
| TeaCache | ✅ | - | ✅ | - | ✅ | ✅ | - |
| FasterCache | - | - | - | - | - | - | ✅ |
| ClusCA | - | ✅ | - | ✅ | - | - | - |
| SpecA | - | ✅ | - | ✅ | - | - | - |

> HunyuanVideo-1.5 标记 ✅ 的方法为官方内置支持（Fora/TaylorSeer/TeaCache），无需额外适配代码。

**附注**

| 目录 | 覆盖的模型变体 |
|------|---------------|
| `flux/` | FLUX.1-dev (原版)、FLUX-Schnell |
| `flux_simple/` | FLUX (Diffusers)、FLUX-NF4、FLUX-LoRA、FLUX-Kontext |
| `qwen/` | Qwen-Image、Qwen-Image-Edit（不同参数） |
| `HunyuanVideo-1.5/` | 官方内置 Fora、TaylorSeer、TeaCache（标记 ✅ 为官方支持，无单独子目录） |

FoRA 实际上是TaylorSeer 中 N = 0 的情况