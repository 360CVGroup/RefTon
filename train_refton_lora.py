import copy, logging, os, json, torch
from datasets_util.datasets_loader import viton_collate_fn
from datasets_util.viton import VITONDataset
from datasets_util.dresscode import DressCodeDataset
from datasets_util.vivid import ViViDDataset
from datasets_util.combined_loader import CombinedDataset
from safetensors.torch import load_file
from pathlib import Path
import numpy as np
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from huggingface_hub import create_repo
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from refton.pipelines import FluxKontextPipelineI2I
from refton.runtime import load_official_diffusers

load_official_diffusers()

import diffusers

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxKontextPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
    parse_buckets_string,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.35.0")

logger = get_logger(__name__)

if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
    )
    return text_encoder_one, text_encoder_two


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError(
                "text_input_ids must be provided when the tokenizer is not specified"
            )

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError(
                "text_input_ids must be provided when the tokenizer is not specified"
            )

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    if hasattr(text_encoders[0], "module"):
        dtype = text_encoders[0].module.dtype
    else:
        dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
    )

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
            "proj_mlp",
        ]

    # now we will add new LoRA weights the transformer layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)

    if args.resume_from_checkpoint == "finetune":
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

        state_dict = load_file(
            f"{os.path.join(args.output_dir, path)}/pytorch_lora_weights.safetensors"
        )
        transformer.load_state_dict(state_dict, strict=False)
        logger.info(
            f"loading_lora_checkpoints_from_{os.path.join(args.output_dir, path)}/pytorch_lora_weights.safetensors"
        )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            modules_to_save = {}
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    modules_to_save["transformer"] = model
                elif isinstance(model, type(unwrap_model(text_encoder_one))):
                    text_encoder_one_lora_layers_to_save = get_peft_model_state_dict(
                        model
                    )
                    modules_to_save["text_encoder"] = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            FluxKontextPipelineI2I.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                **_collate_lora_metadata(modules_to_save),
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = FluxKontextPipelineI2I.lora_state_dict(input_dir)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(
            transformer_, transformer_state_dict, adapter_name="default"
        )
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if args.mixed_precision == "fp16":
            models = [transformer_]
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    def save_model_func(transformer, save_path, args):
        modules_to_save = {}
        transformer = unwrap_model(transformer)
        if args.upcast_before_saving:
            transformer.to(torch.float32)
        else:
            transformer = transformer.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        modules_to_save["transformer"] = transformer

        text_encoder_lora_layers = None

        FluxKontextPipelineI2I.save_lora_weights(
            save_directory=save_path,
            transformer_lora_layers=transformer_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
            **_collate_lora_metadata(modules_to_save),
        )

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(
        filter(lambda p: p.requires_grad, transformer.parameters())
    )

    # Optimization parameters
    transformer_parameters_with_lr = {
        "params": transformer_lora_parameters,
        "lr": args.learning_rate,
    }

    params_to_optimize = [transformer_parameters_with_lr]
    if args.split == "train":
        train = True
    elif args.split == "test":
        train = False
    elif args.split == "all":
        train = None

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if "viton" in args.instance_data_dir:
        train_dataset = VITONDataset(
            args.instance_data_dir,
            instance_prompt=args.instance_prompt,
            scale=args.cond_scale,
            size=(args.height, args.width),
        )
    elif "DressCode" in args.instance_data_dir:
        train_dataset = DressCodeDataset(
            args.instance_data_dir,
            instance_prompt=args.instance_prompt,
            size=(args.height, args.width),
            scale=args.cond_scale,
        )
    elif "ViViD" in args.instance_data_dir:
        train_dataset = ViViDDataset(
            args.instance_data_dir,
            instance_prompt=args.instance_prompt,
            size=(args.height, args.width),
            scale=args.cond_scale,
        )
    else:
        train_dataset = CombinedDataset(
            viton_root=args.instance_data_dir + "/viton",
            dresscode_root=args.instance_data_dir + "/DressCode",
            fashiontryon_root=args.instance_data_dir + "/FashionTryOn_processed",
            vivid_root=args.instance_data_dir + "/vivid_processed",
            igpairs_root=args.instance_data_dir + "/IGPair_processed",
            size=(args.height, args.width),
            instance_prompt=args.instance_prompt,
            scale=args.cond_scale,
            train=train,
        )

    if args.cond_image_column is not None:
        logger.info("I2I fine-tuning enabled.")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,  # 是否打乱数据
        collate_fn=lambda examples: viton_collate_fn(examples),
        num_workers=args.dataloader_num_workers,
        drop_last=False,  # 或者 True，视情况而定
    )

    if not args.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                    text_encoders, tokenizers, prompt, args.max_sequence_length
                )
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
                text_ids = text_ids.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds, text_ids

    if not args.train_text_encoder and not train_dataset.custom_instance_prompts:
        (
            instance_prompt_hidden_states,
            instance_pooled_prompt_embeds,
            instance_text_ids,
        ) = compute_text_embeddings(args.instance_prompt, text_encoders, tokenizers)

    # Clear the memory here
    if not args.train_text_encoder and not train_dataset.custom_instance_prompts:
        text_encoder_one.cpu(), text_encoder_two.cpu()
        del text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
        free_memory()

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.

    if not train_dataset.custom_instance_prompts:
        if not args.train_text_encoder:
            prompt_embeds = instance_prompt_hidden_states
            pooled_prompt_embeds = instance_pooled_prompt_embeds
            text_ids = instance_text_ids

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)

    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes

    num_training_steps_for_scheduler = (
        len(train_dataloader) * accelerator.num_processes * args.num_train_epochs
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        tracker_name = "dreambooth-flux-kontext-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    args.max_train_steps = len(train_dataloader) * args.num_train_epochs

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Training Shape = {train_dataset.size}")
    global_step = 0
    first_epoch = 0

    num_update_steps_per_epoch = len(train_dataloader)
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint and args.resume_from_checkpoint != "finetune":
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.info(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def process_latent_cond(batch, dropout_ref=[0.0, 0.0, 0.0]):
        def pack(x):
            return FluxKontextPipelineI2I._pack_latents(
                x,
                batch_size=x.shape[0],
                num_channels_latents=x.shape[1],
                height=x.shape[2],
                width=x.shape[3],
            )

        def make_ids(x, cond_scale=1.0):
            return FluxKontextPipelineI2I._prepare_latent_image_ids(
                x.shape[0],
                x.shape[2] // 2,
                x.shape[3] // 2,
                accelerator.device,
                weight_dtype,
                cond_scale,
            )

        # 编码主图像 latent
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
        model_input = vae.encode(pixel_values).latent_dist.mode()
        model_input = (
            (model_input - vae_config_shift_factor) * vae_config_scaling_factor
        ).to(dtype=weight_dtype)
        latent_image_ids = make_ids(model_input)
        noise = torch.randn_like(model_input)
        # 编码多个条件图像 latent

        prob_use_person = torch.rand(1)
        if prob_use_person < args.person_prob:
            key_to_index_scale = {
                "cond_pixel_values_person": [1, 1],
            }
        else:
            key_to_index_scale = {
                "cond_pixel_values_agnostic": [1, 1],
            }

        key_to_index_scale["cond_pixel_values_cloth"] = [2, 1]

        prob = torch.rand(3)
        if ("cond_pixel_values_agnostic_mask" in batch) and (prob[0] < dropout_ref[0]):
            key_to_index_scale["cond_pixel_values_agnostic_mask"] = [3, args.cond_scale]
        if ("cond_pixel_values_dense" in batch) and (prob[1] < dropout_ref[1]):
            key_to_index_scale["cond_pixel_values_dense"] = [4, args.cond_scale]
        if ("pixel_values_ref" in batch) and (prob[2] < dropout_ref[2]):
            key_to_index_scale["pixel_values_ref"] = [5, args.cond_scale]
        packed_cond_inputs = []
        cond_latents_ids = []

        for i, key in enumerate(key_to_index_scale):
            cond_pixel = batch[key].to(dtype=vae.dtype)
            cond_latent = vae.encode(cond_pixel).latent_dist.mode()
            cond_latent = (
                (cond_latent - vae_config_shift_factor) * vae_config_scaling_factor
            ).to(dtype=weight_dtype)
            cond_latents_idx = make_ids(cond_latent, key_to_index_scale[key][1])
            cond_latents_idx[..., 0] = key_to_index_scale[key][0]
            cond_latents_ids.append(cond_latents_idx)
            packed_cond_latent = pack(cond_latent)
            packed_cond_inputs.append(packed_cond_latent)

        packed_cond_inputs = torch.cat(packed_cond_inputs, dim=1)

        # latent image IDs
        latent_image_ids = torch.cat([latent_image_ids, *cond_latents_ids], dim=0)

        # 采样噪声并加噪
        bsz = model_input.shape[0]
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=bsz,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )
        indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = noise_scheduler_copy.timesteps[indices].to(
            device=model_input.device
        )
        sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)

        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
        packed_noisy_model_input = pack(noisy_model_input)
        packed_input = torch.cat([packed_noisy_model_input, packed_cond_inputs], dim=1)
        return (
            model_input,
            packed_input,
            latent_image_ids,
            packed_noisy_model_input.shape,
            timesteps,
            noise,
            sigmas,
        )

    has_guidance = unwrap_model(transformer).config.guidance_embeds
    dropout = args.dropout_reference

    all_step_losses = []

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        progress_bar = tqdm(
            range(len(train_dataloader)),
            desc=f"Epoch {epoch}",
            disable=not accelerator.is_local_main_process,
        )

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                # Convert images to latent space
                (
                    model_input,
                    packed_noisy_model_input,
                    latent_image_ids,
                    orig_inp_shape,
                    timesteps,
                    noise,
                    sigmas,
                ) = process_latent_cond(batch, [dropout, dropout, dropout])
                # dsfsdf

                guidance = None
                if has_guidance:
                    guidance = torch.tensor(
                        [args.guidance_scale], device=accelerator.device
                    )
                    guidance = guidance.expand(model_input.shape[0])

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                model_pred = model_pred[:, : orig_inp_shape[1]]
                model_pred = FluxKontextPipelineI2I._unpack_latents(
                    model_pred,
                    height=model_input.shape[2] * vae_scale_factor,
                    width=model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )

                # flow matching loss
                target = noise - model_input

                # Compute regular loss.
                loss = torch.mean(
                    (
                        weighting.float() * (model_pred.float() - target.float()) ** 2
                    ).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                all_step_losses.append(loss.detach().item())

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "global_step": global_step,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        if accelerator.is_main_process:
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            save_model_func(transformer, save_path, args)
            # accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")

            loss_file = os.path.join(args.output_dir, "all_losses.json")
            with open(loss_file, "w") as f:
                json.dump(all_step_losses, f)
            logger.info(f"Updated step losses to {loss_file}")

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # modules_to_save = {}
        # transformer = unwrap_model(transformer)
        # if args.upcast_before_saving:
        #     transformer.to(torch.float32)
        # else:
        #     transformer = transformer.to(weight_dtype)
        # transformer_lora_layers = get_peft_model_state_dict(transformer)
        # modules_to_save["transformer"] = transformer

        # text_encoder_lora_layers = None

        # FluxKontextPipelineI2I.save_lora_weights(
        #     save_directory=args.output_dir,
        #     transformer_lora_layers=transformer_lora_layers,
        #     text_encoder_lora_layers=text_encoder_lora_layers,
        #     **_collate_lora_metadata(modules_to_save),
        # )
        save_model_func(transformer, save_path=args.output_dir, args=args)

        # Final inference
        # Load previous pipeline
        transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            revision=args.revision,
            variant=args.variant,
        )
        pipeline = FluxKontextPipelineI2I.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=transformer,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        # load attention processors
        pipeline.load_lora_weights(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    from argparser import parse_args

    args = parse_args()
    main(args)
