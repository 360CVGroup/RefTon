import json
from unittest import result
from eval import eval_func
import os
import torch
from cleanfid import fid as FID
from PIL import Image
from torch.utils.data import Dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from utils import scan_files_in_dir
from prettytable import PrettyTable
if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_folder_base", type=str, required=True, default="/home/jovyan/liushanyuan-sh-ceph/project/sub_project/gongyue/try-on/datasets/DressCode_gt_test")
    parser.add_argument("--pred_folder_base", type=str, required=True)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()
    cloth_list = ['dresses', 'upper_body', 'lower_body', 'all']
    result_dict = {}
    for cloth in cloth_list:
        pred_folder = os.path.join(args.pred_folder_base, cloth)
        pred_sample = os.listdir(pred_folder)[0]
        img = Image.open(os.path.join(pred_folder, pred_sample))
        
        gt_folder = os.path.join(args.gt_folder_base, f"{cloth}" if img.height == 1024 else f"{cloth}_512")
        result = eval_func( pred_folder=pred_folder,
                                            gt_folder=gt_folder,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            paired=args.paired
                                        )
        result_dict[cloth] = {}
        result_dict[cloth]['result'] = result
        result_dict[cloth]['resolusion'] = img.height
    with open(os.path.join(args.pred_folder_base, 'result.json'), 'w') as f:
        print(f"result in {os.path.join(args.pred_folder_base, 'result.json')}")
        json.dump(result_dict, f, indent=2)