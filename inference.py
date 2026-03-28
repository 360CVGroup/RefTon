import os
import shutil

import torch
from refton.pipelines import FluxKontextPipelineI2I
from train_refton_lora import viton_collate_fn
from datasets_util.viton import VITONDataset
from datasets_util.dresscode import DressCodeDataset
from datasets_util.in_the_wild import MyDataset
from datasets_util.vivid import ViViDDataset
from PIL import Image
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch.distributed as dist


def save_tensor_as_png(tensor, filename="visualize"):
    tensor = tensor.detach().cpu()
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    array = tensor.float().numpy()
    array = np.transpose(array, (1, 2, 0))
    array = (array * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(array).save(filename)


def copy_to_all(src_dir, all_dir):
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

    for f in tqdm(files, desc="Copying files"):
        shutil.copy2(os.path.join(src_dir, f), os.path.join(all_dir, f))


def main(args):

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    # -------------------- Dataset --------------------

    if args.split == "train":
        train = True
    elif args.split == "test":
        train = False
    elif args.split == "all":
        train = None

    if "viton" in args.instance_data_dir:
        dataset = VITONDataset(
            args.instance_data_dir,
            instance_prompt=args.instance_prompt,
            scale=args.cond_scale,
            size=(args.height, args.width),
            train=train, 
            use_different=args.use_different,
        )
    elif "DressCode" in args.instance_data_dir:
        dataset = DressCodeDataset(
            args.instance_data_dir,
            instance_prompt=args.instance_prompt,
            size=(args.height, args.width),
            scale=args.cond_scale,
            train=train,
            use_different=args.use_different,
        )
    else:
        dataset = MyDataset(
                args.instance_data_dir,
                instance_prompt=args.instance_prompt,
                scale=args.cond_scale,
                size=(args.height, args.width)
            )



    dataloader = DataLoader(
        dataset,
        batch_size=args.inference_batch_size,
        shuffle=False,  # 推理不打乱
        collate_fn=lambda examples: viton_collate_fn(examples),
        num_workers=args.dataloader_num_workers,
        drop_last=False,
    )

    # -------------------- Pipeline --------------------
    pipe = FluxKontextPipelineI2I.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=torch.bfloat16
    )
    pipe.load_lora_weights(args.output_dir)

    pipe, dataloader = accelerator.prepare(pipe, dataloader)
    pipe.to(accelerator.device)

    if accelerator.is_main_process:
        print(f"Device: {pipe.device}, Total batches: {len(dataloader)}")

    total_generated_images = 0

    if args.use_person:
        folder_name = "sample_person"
        if args.use_different:
            folder_name += "_unpair"
        key_to_index_scale = {
            "cond_pixel_values_person": [1, 1],
            # "cond_pixel_values_cloth": [2, 1],
        }
        if args.use_reference:
            folder_name += "_ref"
            key_to_index_scale["pixel_values_ref"] = [5, args.cond_scale]
    else:
        folder_name = "sample_agnostic"
        if args.use_different:
            folder_name += "_unpair"

        key_to_index_scale = {
            "cond_pixel_values_agnostic": [1, 1],
            # "cond_pixel_values_cloth": [2, 1],
        }
        if args.use_reference:
            folder_name += "_ref"
            key_to_index_scale["pixel_values_ref"] = [5, args.cond_scale]

    accelerator.wait_for_everyone()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc=f"generating...")):
            images = pipe(
                image=batch,
                batch_size=len(batch["cond_pixel_values_cloth"]),
                prompt=args.instance_prompt,
                num_images_per_prompt=1,
                guidance_scale=2.5,
                generator=torch.Generator().manual_seed(42),
                height=args.height,
                width=args.width,
                cond_scale=args.cond_scale,
                key_to_index_scale=key_to_index_scale,
            ).images
            accelerator.wait_for_everyone()

            images = accelerator.gather(images).to("cpu")
            index = batch["index"]
            gathered_index = index
            if "category" in batch.keys():
                category = batch["category"]
                gathered_category = category
            if torch.distributed.is_initialized():
                gathered_index = [None for _ in range(dist.get_world_size())]
                gathered_category = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered_index, index)

                gathered_index = [
                    item for sublist in gathered_index for item in sublist
                ]
                if "category" in batch.keys():
                    dist.all_gather_object(gathered_category, category)
                    gathered_category = [
                        item for sublist in gathered_category for item in sublist
                    ]

            for i in range(len(images)):

                if "viton" in args.instance_data_dir:
                    if not os.path.exists(f"{args.instance_data_dir}/{folder_name}"):
                        os.makedirs(
                            f"{args.instance_data_dir}/{folder_name}",
                            exist_ok=True,
                        )
                    save_tensor_as_png(
                        images[i],
                        f"{args.instance_data_dir}/{folder_name}/{gathered_index[i]}",
                    )
                elif "DressCode" in args.instance_data_dir:
                    if not os.path.exists(
                        f"{args.instance_data_dir}/{folder_name}/{gathered_category[i]}"
                    ):
                        os.makedirs(
                            f"{args.instance_data_dir}/{folder_name}/{gathered_category[i]}",
                            exist_ok=True,
                        )
                    save_tensor_as_png(
                        images[i],
                        f"{args.instance_data_dir}/{folder_name}/{gathered_category[i]}/{gathered_index[i]}",
                    )
                else:
                    if not os.path.exists(f"{args.instance_data_dir}/{folder_name}"):
                        os.makedirs(
                            f"{args.instance_data_dir}/{folder_name}",
                            exist_ok=True,
                        )
                    save_tensor_as_png(
                        images[i],
                        f"{args.instance_data_dir}/{folder_name}/{gathered_index[i]}",
                    )
                total_generated_images = total_generated_images + 1

            if accelerator.is_main_process:
                print(
                    f"----------------Generated {total_generated_images} images with shape of {list(images.shape)}----------------"
                )


    if "DressCode" in args.instance_data_dir:
        os.makedirs(
            f"{args.instance_data_dir}/{folder_name}/all",
            exist_ok=True,
        )
        copy_to_all(
            f"{args.instance_data_dir}/{folder_name}/upper_body",
            f"{args.instance_data_dir}/{folder_name}/all",
        )
        copy_to_all(
            f"{args.instance_data_dir}/{folder_name}/lower_body",
            f"{args.instance_data_dir}/{folder_name}/all",
        )
        copy_to_all(
            f"{args.instance_data_dir}/{folder_name}/dresses",
            f"{args.instance_data_dir}/{folder_name}/all",
        )


if __name__ == "__main__":
    from argparser import parse_args

    args = parse_args()
    main(args)
