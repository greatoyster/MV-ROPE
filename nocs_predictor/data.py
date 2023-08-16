import os
import cv2
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader

try:
    import ipdb as pdb
except:
    import pdb
from tqdm import tqdm

from joblib import Memory
import pandas as pd

# Create a memory object with cache directory
cache_dir = "__pycache__"  # Specify your cache directory
memory = Memory(location=cache_dir, verbose=0)


# @memory.cache
def cached_init(datapath, mode):
    color_paths_final = []
    coord_paths_final = []
    mask_paths_final = []
    meta_paths_final = []

    color_suffix = "*_color.png"
    coord_suffix = "*_coord.png"
    mask_suffix = "*_mask.pred.png"
    meta_suffix = "*_meta.pred.txt"
    bad_suffix = "*_bad.png"

    assert mode in ["train", "val", "test"]

    sub_folders = sorted(os.listdir(os.path.join(datapath, mode)))
    for sf in tqdm(sub_folders):
        color_path_pattern = os.path.join(datapath, mode, sf, color_suffix)
        coord_path_pattern = os.path.join(datapath, mode, sf, coord_suffix)
        mask_path_pattern = os.path.join(datapath, mode, sf, mask_suffix)
        meta_path_pattern = os.path.join(datapath, mode, sf, meta_suffix)
        bad_path_pattern = os.path.join(datapath, mode, sf, bad_suffix)

        color_paths = sorted(glob(color_path_pattern))
        coord_paths = sorted(glob(coord_path_pattern))
        mask_paths = sorted(glob(mask_path_pattern))
        meta_paths = sorted(glob(meta_path_pattern))
        bad_paths = sorted(glob(bad_path_pattern), reverse=True)

        # check bad image path
        for bp in bad_paths:
            bad_id = int(os.path.basename(bp)[:4])
            coord_paths.pop(bad_id)
            mask_paths.pop(bad_id)
            meta_paths.pop(bad_id)

        color_paths_final += color_paths
        coord_paths_final += coord_paths
        mask_paths_final += mask_paths
        meta_paths_final += meta_paths

    return color_paths_final, coord_paths_final, mask_paths_final, meta_paths_final


class NOCSDataset(Dataset):
    MEAN_PIXEL = np.array([[120.66209412, 114.70348358, 105.81269836]])
    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640

    def __init__(self, datapath, mode):
        self.datapath = datapath
        (
            self.color_paths,
            self.coord_paths,
            self.mask_paths,
            self.meta_paths,
        ) = cached_init(datapath, mode)
        self.mode = mode
        print(len(self.color_paths))
        print(len(self.coord_paths))
        print(len(self.mask_paths))
        print(len(self.meta_paths))
        if mode == "test":
            assert (
                len(self.color_paths) == len(self.mask_paths) == len(self.meta_paths)
            ), "data mismatch"
        else:
            assert (
                len(self.color_paths)
                == len(self.coord_paths)
                == len(self.mask_paths)
                == len(self.meta_paths)
            ), "data mismatch"

    def __len__(self):
        return len(self.color_paths)

    def __getitem__(self, idx):
        # Return the normalized object coordinates as a tensor
        color = self.color_paths[idx]
        mask = self.mask_paths[idx]
        meta = self.meta_paths[idx]

        color_tensor = cv2.imread(color, cv2.IMREAD_ANYCOLOR)
        color_tensor = color_tensor[:, :, [2, 1, 0]]  # BGR to RGB
        mask_tensor = cv2.imread(mask)[:, :, 2]
        mask_tensor = torch.as_tensor(mask_tensor).long()

        color_tensor = color_tensor.astype(np.float32) / 255.0
        color_tensor = torch.as_tensor(color_tensor).permute(2, 0, 1)

        if self.mode != "test":
            coord = self.coord_paths[idx]
            coord_tensor = cv2.imread(coord, cv2.IMREAD_ANYCOLOR)
            coord_tensor = coord_tensor.astype(np.float32) / 255.0
            coord_tensor = torch.as_tensor(coord_tensor).permute(2, 0, 1)

        meta_match = torch.as_tensor(
            pd.read_csv(meta, sep=" ", header=None).to_numpy()[:, 1].astype(np.int32)
        )
        pad_tensor = torch.zeros((25 - meta_match.shape[0]), dtype=meta_match.dtype)
        meta_match = torch.cat((meta_match, pad_tensor), dim=0)

        # process image_tensor
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=color_tensor.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=color_tensor.device)
        color_tensor = color_tensor.sub_(mean[:, None, None]).div_(std[:, None, None])

        if self.mode == "test":
            name_split = color.split("/")
            scene_id = name_split[-2]
            frame_id = name_split[-1].split("_")[0]
            return color_tensor, mask_tensor, meta_match, scene_id, frame_id
        else:
            return color_tensor, mask_tensor, coord_tensor, meta_match
