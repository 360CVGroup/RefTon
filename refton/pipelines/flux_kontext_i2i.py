import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)

from refton.runtime.official_diffusers import load_official_diffusers

load_official_diffusers()

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FluxIPAdapterMixin,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor


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
        >>> from refton.pipelines import FluxKontextPipelineI2I
        >>> from diffusers.utils import load_image

        >>> pipe = FluxKontextPipelineI2I.from_pretrained(
        ...     "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
        ... )
        >>> pipe.to("cuda")

        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yarn-art-pikachu.png"
        ... ).convert("RGB")
        >>> prompt = "Make Pikachu hold a sign that says 'Black Forest Labs is awesome', yarn art style, detailed, vibrant colors"
        >>> image = pipe(
        ...     image=image,
        ...     prompt=prompt,
        ...     guidance_scale=2.5,
        ...     generator=torch.Generator().manual_seed(42),
        ... ).images[0]
        >>> image.save("output.png")
        ```
"""

PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


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
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class FluxKontextPipelineI2I(
    DiffusionPipeline,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
    FluxIPAdapterMixin,
):
    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->transformer->vae"
    _optional_components = ["image_encoder", "feature_extractor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )
        self.latent_channels = (
            self.vae.config.latent_channels if getattr(self, "vae", None) else 16
        )
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length
            if hasattr(self, "tokenizer") and self.tokenizer is not None
            else 77
        )
        self.default_sample_size = 128

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer_2.batch_decode(
                untruncated_ids[:, self.tokenizer_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(
            text_input_ids.to(device), output_hidden_states=False
        )[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), output_hidden_states=False
        )

        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        device = device or self._execution_device

        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = (
            self.text_encoder.dtype
            if self.text_encoder is not None
            else self.transformer.dtype
        )
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def encode_image(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder(image).image_embeds
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        return image_embeds

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype, scale=1):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height)[:, None]
        ) * scale + scale // 2
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width)[None, :]
        ) * scale + scale // 2

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = (
            latent_image_ids.shape
        )

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(
            batch_size, num_channels_latents, height // 2, 2, width // 2, 2
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size, (height // 2) * (width // 2), num_channels_latents * 4
        )

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        generator: torch.Generator,
        device,
        dtype,
        key_to_index_scale: dict,
    ):
        image_latents, image_ids = [], []
        for key in key_to_index_scale.keys():
            image_latent = image[key].to(device=device, dtype=dtype)
            image_latent = self.vae.encode(image_latent).latent_dist.mode()
            image_latent = (
                image_latent - self.vae.config.shift_factor
            ) * self.vae.config.scaling_factor
            image_latent_height, image_latent_width = image_latent.shape[2:]
            image_idx = self._prepare_latent_image_ids(
                image_latent[0],
                image_latent.shape[2] // 2,
                image_latent.shape[3] // 2,
                device,
                dtype,
                key_to_index_scale[key][1],
            )
            image_idx[..., 0] = key_to_index_scale[key][0]
            image_latent = self._pack_latents(
                image_latent,
                image_latent.shape[0],
                image_latent.shape[1],
                image_latent_height,
                image_latent_width,
            )
            print(key, image[key].shape, image_idx[0:5,:])
            image_ids.append(image_idx)
            image_latents.append(image_latent)

        return image_latents, image_ids

    def prepare_latents(
        self,
        image: Optional[torch.Tensor],
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        key_to_index_scale: dict,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, height, width)

        image_latents = image_ids = None

        if image is not None:
            image_latents, image_ids = self._encode_vae_image(
                image=image,
                generator=generator,
                device=device,
                dtype=dtype,
                key_to_index_scale=key_to_index_scale,
            )
            image_latents = torch.cat(image_latents, dim=1)
        latent_ids = self._prepare_latent_image_ids(
            batch_size, height // 2, width // 2, device, dtype
        )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(
            latents, batch_size, num_channels_latents, height, width
        )

        return latents, image_latents, latent_ids, image_ids

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    def visualize(self, latents, height, width, output_type, t):
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)
        image[0].save(f"process/{t}_output.png")

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        batch_size: int = 1,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        cond_scale: int = 2.0,
        key_to_index_scale=None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 2,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        max_area: int = 1024**2,
        _auto_resize: bool = True,
    ):
        """
        Generate images with the Flux Kontext image-to-image pipeline.

        Examples:
        """
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None)
            if self.joint_attention_kwargs is not None
            else None
        )

        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents, latent_ids, image_ids = self.prepare_latents(
            image,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
            key_to_index_scale=key_to_index_scale,
        )
        if image_ids is not None:
            latent_ids = torch.cat([latent_ids, *image_ids], dim=0)

        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None

        self.scheduler.set_begin_index(0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = (
                        image_embeds
                    )

                latent_model_input = latents
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, : latents.size(1)]

                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image * 0.5 + 0.5).clamp(0, 1)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)


__all__ = ["FluxKontextPipelineI2I"]