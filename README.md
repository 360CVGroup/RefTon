
  

# RefTon: Reference person shot assist virtual Try-On

  

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-RefVTON-ff9900?style=flat)](https://huggingface.co/qihoo360/RefVTON) [![arXiv](https://img.shields.io/badge/arXiv-2511.00956-B31B1B?style=flat)](https://arxiv.org/abs/2511.00956)

  

![examples](assets/examples.png)

  

  

We propose **RefVTON**, an Person-to-Person Virtual Try-on model with Additional Visual Reference, that directly fits the target garment onto the person image while incorporating reference images to enhance the model's ability to preserve and accurately depict clothing details.

  

  

  

## 💡 Update

  

  

- [x] [2025.10.11] Release the virtual try-on inference code and LoRA weights.

  

  

- [x] [2025.10.13] Release the technical report on Arxiv.

  

  

  

## 💪 Highlight Feature

  

  

  

- **End-to-End Virtual Try-On Model:**  Adapted from [Flux-Kontext](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/flux), our model can operate either as an inpainting system that transfer target garments into masked regions or as a direct garment-transfer model that applies the clothing onto the human body.

- **Reference-Guided Try-On for Enhanced Realism:**  To reflect users’ preference for realistic wearing effects rather than isolated garment appearance in online shopping, our model supports using reference images of a person wearing the target clothing. This enables better preservation of material textures and fine design details.

- **Improved Performance:**  Our model achieves state-of-the-art results on public benchmarks and demonstrates strong generalization to in-the-wild inputs.


  

  

  

## 🧩 Environment Setup

  

  

  

```
conda create -n RefVTON python=3.12 -y
conda activate RefVTON
pip install -r requirements.txt
cd diffusers
pip install -e .
cd ..
```

  

  

  

## 📂 Preparation of Dataset and Pretrained Models

  

  

  

### Dataset

  

  

  

Currently, we provide a small test set that includes additional reference images of **different persons wearing the target clothes** for testing our model. You can find it in the `examples_try` folder. We plan to release the reference data generation code, along with our complete dataset containing model reference images, in the future.


  

  

Nevertheless, inference can still be performed in a reference-free setting on public benchmarks, including [VITON-HD](https://github.com/shadow2496/VITON-HD) and [DressCode](https://github.com/aimagelab/dress-code).

  

### Pretrained Models

  

  

We provide pretrained backbone networks and LoRA weights for evaluation and deployment. Please download the `.safetensors` files from [here](https://huggingface.co/qihoo360/RefVTON) and place them in the `checkpoints` directory. In addition, download the pretrained weights of **Flux-Kontext.dev** from [here](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) to serve as the backbone model.

  

  

  

## ⏳ Inference Pipeline

  

  

  

Here we provide the inference code for our RefVTON.

  

  

```
accelerate launch --num_processes 8 --main_process_port 29500 inference.py \
--pretrained_model_name_or_path="[path_to_your_Flux_model]" \
--instance_data_dir="[your_data_directory]" \ # e.g. "./example" can be used as a demo directory
--checkpoint_weight="[Path_to_LoRA_weights]" \
--mixed_precision="bf16" \
--split="test" \
--height=512 \
--width=384 \
--inference_batch_size=1 \
--cond_scale=1 \
--seed="0" \
--use_reference \
--use_different \
--use_person
```

  

  

-  `pretrained_model_name_or_path`: Path to the downloaded Flux-Kontext model weights.

  

  

-  `instance_data_dir`: Path to your dataset. If you want to inference on VITON-HD or DressCode, ensure that the words "viton" or "DressCode" appear in the path.
If you want to use your own dataset, you can organize your files following the directory structure provided in our `./example`, and then set this argument to the corresponding path.
  

  

-  `checkpoint_weight`: Path to the downloaded or trained LoRA weights. Make sure to select the weights that match your desired resolution — for example,
`512_384_pytorch_lora_weights.safetensors` corresponds to a resolution of $512\times384$ 

  

  

-  `cond_scale`: Resize scale of the reference image during training. Defaults to `1.0` for $512\times384$ and `2.0` for $1024\times768$ resolution.

  

  

-  `use_reference`: Whether to use a additonal reference image as input.

  

  

-  `use_different`: **Only applicable for VITON/DressCode inference.** Whether to use different cloth-person pairs.

  

  

-  `use_person`: **Only applicable for VITON/DressCode inference.** Whether to use the unmasked person image instead of the agnostic masked image as input for the virtual try-on task.

  
We conduct inference on an A800 GPU, which requires approximately 34 GB GPU memory.
Each image takes about 10 seconds to generate, and the resulting images will be saved automatically in the directory specified by instance_data_dir (e.g. `example/sample_person_unpair_ref`).
  

## 📊 Evaluation

  

  

We quantitatively evaluate the quality of virtual try-on results using the FID, KID, SSIM, and LPIPS. Here, we provide the evaluation code for the VITON-HD and DressCode datasets.

  

```
# Evaluation on VITON-HD dataset
CUDA_VISIBLE_DEVICES=0 python eval_dresscode.py \
--gt_folder_base [path_to_your_ground_truth_image_folder] \
--pred_folder_base [[path_to_your_generated_image_folder]]\
--paired
```

  

  

  

```
# Evaluation on DressCode dataset

CUDA_VISIBLE_DEVICES=0 python eval.py \
--gt_folder_base [path_to_your_ground_truth_image_folder] \
--pred_folder_base [[path_to_your_generated_image_folder]]\
```

  

  

-  `paired`: If you perform unpaired generation, where different garments are fitted onto the target person, you should enable this flag during evaluation.

  
  

Evaluation result on VITON-HD and DressCode dataset:

![examples](assets/result.png)

  
  

Evaluation result on three subsets of DressCode dataset:

![examples](assets/dresscode_subset.png)

  

## 🌸 Acknowledgement

  

  

This code is mainly built upon [Diffusers](https://github.com/huggingface/diffusers/tree/main), [Flux](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/flux), and [CatVTON](https://github.com/Zheng-Chong/CatVTON/) repositories. Thanks so much for their solid work!

  

  

  

## 💖 Citation


If you find this repository useful, please consider citing our paper:
```
@misc{li2025refvtonpersontopersontryadditional,
      title={RefVTON: person-to-person Try on with Additional Unpaired Visual Reference}, 
      author={Liuzhuozheng Li and Yue Gong and Shanyuan Liu and Bo Cheng and Yuhang Ma and Liebucha Wu and Dengyang Jiang and Zanyi Wang and Dawei Leng and Yuhui Yin},
      year={2025},
      eprint={2511.00956},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.00956}, 
}
```
