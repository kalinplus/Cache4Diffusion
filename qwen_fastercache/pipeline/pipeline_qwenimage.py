# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# FasterCache adaptation for QwenImagePipeline.
#
# Key differences from freqca/pipeline/pipeline_qwenimage.py:
#   - `cache_dic` / `current` parameters are REMOVED.
#   - A plain integer `counter` (denoising step index) is used instead.
#   - New parameters: `fastercache_start_step`, `fastercache_interval`,
#     `fastercache_alpha` control the caching schedule.
#   - `transformer.reset_fastercache()` is called at the start of each
#     denoising run to clear per-block caches from the previous image.
#   - The denoising loop is the standard `for i, t in enumerate(timesteps)`,
#     not the freqca `for _ in range(8 * num_steps)` variant.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import QwenImageLoraLoaderMixin
from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from pipeline.pipeline_qwenimage import QwenImagePipeline
        >>> from fastercache_utils import pipeline_with_fastercache

        >>> pipe = QwenImagePipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> pipe = pipeline_with_fastercache(pipe)  # enable FasterCache

        >>> image = pipe("A cat holding a sign that says hello world").images[0]
        >>> image.save("qwenimage_fastercache.png")
        ```
"""


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(f"The current scheduler class {scheduler.__class__} does not support custom timestep schedules.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(f"The current scheduler class {scheduler.__class__} does not support custom sigmas schedules.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class QwenImagePipeline(DiffusionPipeline, QwenImageLoraLoaderMixin):
    """
    Qwen text-to-image pipeline with FasterCache support.

    Compared to the vanilla diffusers pipeline, this version accepts
    FasterCache schedule parameters and passes a step `counter` to the
    transformer so blocks can skip attention computation on cached steps.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLQwenImage,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        transformer: QwenImageTransformer2DModel,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = 1024
        self.prompt_template_encode = (
            "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
            "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )
        self.prompt_template_encode_start_idx = 34
        self.default_sample_size = 128

    # ------------------------------------------------------------------
    # Text encoding helpers (unchanged)
    # ------------------------------------------------------------------

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        template  = self.prompt_template_encode
        drop_idx  = self.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt]
        txt_tokens = self.tokenizer(
            txt, max_length=self.tokenizer_max_length + drop_idx,
            padding=True, truncation=True, return_tensors="pt",
        ).to(device)
        encoder_hidden_states = self.text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 1024,
    ):
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt, device)

        prompt_embeds      = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds      = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds      = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def check_inputs(
        self, prompt, height, width,
        negative_prompt=None,
        prompt_embeds=None, negative_prompt_embeds=None,
        prompt_embeds_mask=None, negative_prompt_embeds_mask=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} "
                f"but are {height} and {width}. Dimensions will be resized accordingly"
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, "
                f"but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`.")
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        elif prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("Cannot forward both `negative_prompt` and `negative_prompt_embeds`.")
        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError("If `prompt_embeds` are provided, `prompt_embeds_mask` must also be provided.")
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError("If `negative_prompt_embeds` are provided, `negative_prompt_embeds_mask` must also be provided.")
        if max_sequence_length is not None and max_sequence_length > 1024:
            raise ValueError(f"`max_sequence_length` cannot be greater than 1024 but is {max_sequence_length}")

    # ------------------------------------------------------------------
    # Latent packing / unpacking helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width  = 2 * (int(width)  // (vae_scale_factor * 2))
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)
        return latents

    def enable_vae_slicing(self):   self.vae.enable_slicing()
    def disable_vae_slicing(self):  self.vae.disable_slicing()
    def enable_vae_tiling(self):    self.vae.enable_tiling()
    def disable_vae_tiling(self):   self.vae.disable_tiling()

    def prepare_latents(
        self, batch_size, num_channels_latents, height, width,
        dtype, device, generator, latents=None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width  = 2 * (int(width)  // (self.vae_scale_factor * 2))
        shape  = (batch_size, 1, num_channels_latents, height, width)

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, "
                f"but requested an effective batch size of {batch_size}."
            )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        return latents

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def guidance_scale(self):    return self._guidance_scale
    @property
    def attention_kwargs(self):  return self._attention_kwargs
    @property
    def num_timesteps(self):     return self._num_timesteps
    @property
    def current_timestep(self):  return self._current_timestep
    @property
    def interrupt(self):         return self._interrupt

    # ------------------------------------------------------------------
    # Main call
    # ------------------------------------------------------------------

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        # ---- FasterCache schedule ----
        fastercache_start_step: Optional[int] = None,
        fastercache_interval: Optional[int] = None,
        fastercache_alpha: Optional[float] = None,
    ):
        r"""
        Generate images with FasterCache-accelerated attention.

        FasterCache parameters
        ----------------------
        fastercache_start_step : int, optional
            Steps 0 … start_step always run full attention (cache warm-up).
            Defaults to the value set in `pipeline_with_fastercache` (15).
        fastercache_interval : int, optional
            Attention is recomputed every `interval` steps; the rest are skipped.
            Defaults to the value set in `pipeline_with_fastercache` (2).
        fastercache_alpha : float, optional
            Linear extrapolation coefficient on skip steps.  0 = plain reuse.
            Defaults to the value set in `pipeline_with_fastercache` (0.3).

        Examples:

        Returns:
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] or `tuple`.
        """
        height = height or self.default_sample_size * self.vae_scale_factor
        width  = width  or self.default_sample_size * self.vae_scale_factor

        # Resolve FasterCache defaults from transformer attributes (set by pipeline_with_fastercache)
        fc_start    = fastercache_start_step if fastercache_start_step is not None \
                      else getattr(self.transformer, "_fc_default_start_step", 15)
        fc_interval = fastercache_interval   if fastercache_interval   is not None \
                      else getattr(self.transformer, "_fc_default_cache_interval", 2)
        fc_alpha    = fastercache_alpha      if fastercache_alpha      is not None \
                      else getattr(self.transformer, "_fc_default_alpha", 0.3)

        # 1. Input validation
        self.check_inputs(
            prompt, height, width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale   = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt        = False

        # 2. Batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )

        if true_cfg_scale > 1 and not has_neg_prompt:
            logger.warning(
                f"true_cfg_scale={true_cfg_scale} but no negative_prompt provided; CFG disabled."
            )
        elif true_cfg_scale <= 1 and has_neg_prompt:
            logger.warning(
                "negative_prompt is provided but true_cfg_scale <= 1; CFG disabled."
            )

        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        # 3. Encode prompts
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        # 4. Prepare latents
        num_channels_latents = self.transformer.config.in_channels // 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents, height, width,
            prompt_embeds.dtype, device, generator, latents,
        )
        img_shapes = [[(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)]] * batch_size

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu,
        )
        self._num_timesteps = len(timesteps)

        # 6. Guidance embedding (guidance-distilled models only)
        if self.transformer.config.guidance_embeds and guidance_scale is None:
            raise ValueError("guidance_scale is required for guidance-distilled model.")
        elif self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens          = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )

        # 7. Clear FasterCache caches from any previous run
        if hasattr(self.transformer, "reset_fastercache"):
            self.transformer.reset_fastercache()

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # FLOPs tracking (mirrors freqca; baseline = full model, 50 steps, 1328×1328)
        _BASELINE_FLOPS_T = 12917.56   # TFLOPs for one full run without any caching
        test_FLOPs = getattr(self, '_test_FLOPs', False)
        if test_FLOPs:
            total_flops = 0.0
            total_macs  = 0.0
            total_params = 0.0

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # shared kwargs for transformer call
                _fc_kwargs = dict(
                    counter=i,
                    fastercache_start_step=fc_start,
                    fastercache_interval=fc_interval,
                    fastercache_alpha=fc_alpha,
                )

                # ---- conditional forward pass ----
                with self.transformer.cache_context("cond"):
                    _cond_inputs = dict(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                        fastercache_module="cond",
                        **_fc_kwargs,
                    )
                    if test_FLOPs:
                        from calflops import calculate_flops
                        flops, macs, params = calculate_flops(model=self.transformer, kwargs=_cond_inputs)
                        total_flops  += float(flops)
                        total_macs   += float(macs)
                        total_params += float(params)
                    noise_pred = self.transformer(**_cond_inputs)[0]

                # ---- unconditional forward pass (true CFG only) ----
                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        _uncond_inputs = dict(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states=negative_prompt_embeds,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            img_shapes=img_shapes,
                            txt_seq_lens=negative_txt_seq_lens,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                            fastercache_module="uncond",
                            **_fc_kwargs,
                        )
                        if test_FLOPs:
                            from calflops import calculate_flops
                            flops, macs, params = calculate_flops(model=self.transformer, kwargs=_uncond_inputs)
                            total_flops  += float(flops)
                            total_macs   += float(macs)
                            total_params += float(params)
                        neg_noise_pred = self.transformer(**_uncond_inputs)[0]

                    # CFG rescaling (same as freqca)
                    comb_pred  = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                    cond_norm  = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred,  dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                # ---- scheduler step ----
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents       = callback_outputs.pop("latents",       latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if test_FLOPs:
            speedup = _BASELINE_FLOPS_T / (total_flops * 1e-12)
            print(f"\n[FasterCache FLOPs Report]")
            print(f"  Total FLOPs : {total_flops * 1e-12:.2f} T")
            print(f"  Speedup     : {speedup:.2f}x  (baseline {_BASELINE_FLOPS_T:.2f} T)")
            print(f"  Total MACs  : {total_macs  * 1e-12:.2f} T")
            print(f"  Params      : {total_params * 1e-9 :.2f} G")

        self._current_timestep = None

        # 9. Decode latents
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean
            image   = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image   = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return QwenImagePipelineOutput(images=image)
