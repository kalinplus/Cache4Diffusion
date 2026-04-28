# Copyright 2025 Qwen-Image Team, The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# FasterCache adaptation for Qwen dual-stream DiT.
#
# Core idea (from FasterCache paper):
#   Adjacent denoising steps produce very similar attention outputs.
#   We cache the img + txt attention results from a "full" step,
#   and reuse (or linearly extrapolate) them on "skip" steps,
#   avoiding the O(n^2) attention computation on those steps.
#
# Ported from:
#   - FasterCache/scripts/vchitect/fastercache_sample_vchitect.py  (dual-stream, cache + reuse)
#   - FasterCache/scripts/cogvideox/fastercache_sample_cogvideox.py (linear extrapolation)
#   - freqca/pipeline/transformer_qwenimage.py                      (Qwen base transformer)

import functools
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import AttentionMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, RMSNorm


logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# RoPE helpers (unchanged from Qwen base)
# ---------------------------------------------------------------------------

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent).to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if use_real:
        cos, sin = freqs_cis
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)
        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
        return x_out.type_as(x)


class QwenTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep, hidden_states):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))
        return timesteps_emb


class QwenEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.rope_cache = {}
        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def forward(self, video_fhw, txt_seq_lens, device):
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"
            if not torch.compiler.is_compiling():
                if rope_key not in self.rope_cache:
                    self.rope_cache[rope_key] = self._compute_video_freqs(frame, height, width, idx)
                video_freq = self.rope_cache[rope_key]
            else:
                video_freq = self._compute_video_freqs(frame, height, width, idx)
            video_freq = video_freq.to(device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index: max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)
        return vid_freqs, txt_freqs

    @functools.lru_cache(maxsize=None)
    def _compute_video_freqs(self, frame, height, width, idx=0):
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx: idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2):], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2):], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


# ---------------------------------------------------------------------------
# Attention processor (unchanged from Qwen base)
# ---------------------------------------------------------------------------

class QwenDoubleStreamAttnProcessor2_0:
    """Joint attention for Qwen dual-stream (image + text)."""

    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("QwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0+")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states")

        seq_txt = encoder_hidden_states.shape[1]

        img_query = attn.to_q(hidden_states)
        img_key   = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key   = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key   = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))
        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key   = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key   = apply_rotary_emb_qwen(img_key,   img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key   = apply_rotary_emb_qwen(txt_key,   txt_freqs, use_real=False)

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key   = torch.cat([txt_key,   img_key],   dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        joint_hidden_states = dispatch_attention_fn(
            joint_query, joint_key, joint_value,
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            backend=self._attention_backend,
        )

        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


# ---------------------------------------------------------------------------
# Transformer Block — FasterCache version
# ---------------------------------------------------------------------------

@maybe_allow_in_graph
class QwenImageTransformerBlock(nn.Module):
    """
    Dual-stream DiT block with FasterCache support.

    FasterCache strategy (block-level attention caching):
      - On "full" steps: compute attention normally, save (img_attn, txt_attn) to cache.
      - On "skip" steps: reuse cached outputs, optionally applying linear extrapolation
        with coefficient `_fc_alpha` using the last two cached values.

    Skip condition (following CogVideoX FasterCache convention):
        counter > _fc_start_step  AND  counter % _fc_interval != 0

    Cache is keyed by `_fc_module` ('cond' or 'uncond') so that CFG mode
    keeps separate caches for the two forward passes.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=QwenDoubleStreamAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=eps,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # Per-block attention cache; populated at runtime.
        # Structure: { module_key: {'img_new': Tensor|None, 'img_old': Tensor|None,
        #                           'txt_new': Tensor|None, 'txt_old': Tensor|None} }
        self._fc_caches: Dict[str, Dict] = {}

    def _modulate(self, x, mod_params):
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)

    def reset_fastercache(self):
        """Clear all cached attention outputs (call before a new denoising run)."""
        self._fc_caches.clear()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        # ---- FasterCache control knobs ----
        _fc_counter: Optional[int] = None,
        _fc_module: str = "cond",
        _fc_start_step: int = 15,
        _fc_interval: int = 2,
        _fc_alpha: float = 0.3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # ---- modulation (always computed — cheap linear layers) ----
        img_mod_params = self.img_mod(temb)
        txt_mod_params = self.txt_mod(temb)
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        img_normed    = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

        txt_normed    = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        # ---- FasterCache: decide whether to skip attention ----
        # Lazy-init: the dict may not exist when forward is monkey-patched onto a
        # diffusers block instance that was not created by our __init__.
        if not hasattr(self, '_fc_caches'):
            self._fc_caches = {}
        module_cache = self._fc_caches.get(_fc_module, {
            "img_new": None, "img_old": None,
            "txt_new": None, "txt_old": None,
        })
        has_cache = module_cache["img_new"] is not None
        use_cache = (
            _fc_counter is not None
            and has_cache
            and _fc_counter > _fc_start_step
            and _fc_counter % _fc_interval != 0   # skip on non-aligned steps
        )

        if use_cache:
            # Linear extrapolation: out ≈ cached[-1] + α*(cached[-1] - cached[-2])
            # Falls back to plain reuse when only one cached value is available.
            if module_cache["img_old"] is not None:
                img_attn_output = (
                    module_cache["img_new"]
                    + _fc_alpha * (module_cache["img_new"] - module_cache["img_old"])
                )
                txt_attn_output = (
                    module_cache["txt_new"]
                    + _fc_alpha * (module_cache["txt_new"] - module_cache["txt_old"])
                )
            else:
                img_attn_output = module_cache["img_new"]
                txt_attn_output = module_cache["txt_new"]
        else:
            # Full attention computation
            joint_attention_kwargs = joint_attention_kwargs or {}
            attn_output = self.attn(
                hidden_states=img_modulated,
                encoder_hidden_states=txt_modulated,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                image_rotary_emb=image_rotary_emb,
                **joint_attention_kwargs,
            )
            img_attn_output, txt_attn_output = attn_output

            # Shift cache: old ← new, new ← fresh
            self._fc_caches[_fc_module] = {
                "img_old": module_cache["img_new"],
                "img_new": img_attn_output,
                "txt_old": module_cache["txt_new"],
                "txt_new": txt_attn_output,
            }

        # ---- residual connections ----
        hidden_states         = hidden_states         + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # ---- image MLP ----
        img_normed2    = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        hidden_states  = hidden_states + img_gate2 * self.img_mlp(img_modulated2)

        # ---- text MLP ----
        txt_normed2    = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * self.txt_mlp(txt_modulated2)

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


# ---------------------------------------------------------------------------
# Transformer model — FasterCache version
# ---------------------------------------------------------------------------

class QwenImageTransformer2DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    """
    Qwen dual-stream DiT with FasterCache acceleration.

    Key difference from the original diffusers model:
      - `forward` accepts `counter` (int, denoising step index 0-based) instead of
        the freqca-style `cache_dic` / `current` dicts.
      - FasterCache config (`fastercache_start_step`, `fastercache_interval`,
        `fastercache_alpha`, `fastercache_module`) is forwarded to every block.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["QwenImageTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["QwenImageTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)
        self.time_text_embed = QwenTimestepProjEmbeddings(embedding_dim=self.inner_dim)
        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)
        self.img_in  = nn.Linear(in_channels,       self.inner_dim)
        self.txt_in  = nn.Linear(joint_attention_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)
        self.gradient_checkpointing = False

    def reset_fastercache(self):
        """Clear all per-block attention caches. Call before each new image generation."""
        for block in self.transformer_blocks:
            if hasattr(block, '_fc_caches'):
                block._fc_caches.clear()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        return_dict: bool = True,
        # ---- FasterCache parameters (replace cache_dic / current) ----
        counter: Optional[int] = None,
        fastercache_module: str = "cond",
        fastercache_start_step: int = 15,
        fastercache_interval: int = 2,
        fastercache_alpha: float = 0.3,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        Args:
            hidden_states: Packed image latent tokens [B, S, C].
            encoder_hidden_states: Text embeddings [B, T, D].
            encoder_hidden_states_mask: Text attention mask [B, T].
            timestep: Denoising timestep [B].
            img_shapes: List of (frame, height, width) per image in the batch.
            txt_seq_lens: Valid text token lengths per image.
            guidance: Optional guidance scale tensor (guidance-distilled models).
            attention_kwargs: Passed to the attention processor.
            counter: Denoising step index (0-based). Used by FasterCache to decide
                     whether to skip attention on the current step.
            fastercache_module: 'cond' or 'uncond'. Keeps separate caches for the
                                 two CFG forward passes.
            fastercache_start_step: Steps with index <= this value always run full
                                     attention (cache warm-up phase).
            fastercache_interval: Every `fastercache_interval`-th step (0, interval,
                                   2*interval, …) runs full attention; others are skipped.
            fastercache_alpha: Linear extrapolation coefficient for skipped steps.
                               0.0 = plain reuse; 0.3 = mild extrapolation (default).
        """
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )

        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                # Gradient checkpointing does not forward FasterCache kwargs to avoid
                # recomputation graph issues; blocks run with full attention in this mode.
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_mask,
                    temb,
                    image_rotary_emb,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                    # FasterCache knobs
                    _fc_counter=counter,
                    _fc_module=fastercache_module,
                    _fc_start_step=fastercache_start_step,
                    _fc_interval=fastercache_interval,
                    _fc_alpha=fastercache_alpha,
                )

            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
