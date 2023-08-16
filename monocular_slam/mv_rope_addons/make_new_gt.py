"""
We use sam segmentation instead of ground truth label, so we need to recompute new associated ground truth  
"""

from tqdm import tqdm
import numpy as np
import torch
import cv2
import argparse
import pickle
from glob import glob
import sys
import os

# fmt: off
sys.path.append('droid_slam')
import geom.utils as gutils
# fmt: on

parser = argparse.ArgumentParser()
parser.add_argument(
    "--datapath", default="/mnt/wd8t/Datasets/DROID_DATA/nocs/real/real_test"
)

args = parser.parse_args()
datapath = args.datapath
scenes = ["scene_1", "scene_2", "scene_3", "scene_4", "scene_5", "scene_6"]
for scene in scenes:
    masks = sorted(glob(os.path.join(datapath, scene, "*_mask.pred.png")))
    depths = sorted(glob(os.path.join(datapath, scene, "*_depth.png")))
    nocss = sorted(glob(os.path.join(datapath, scene, "*_coord.png")))
    metas = sorted(glob(os.path.join(datapath, scene, "*_meta.pred.txt")))

    assert len(depths) == len(masks)
    assert len(nocss) == len(masks)
    assert len(metas) == len(masks)

    fx, fy, cx, cy = 591.0125, 590.16775, 322.525, 244.11084

    bar = tqdm(zip(masks, depths, nocss, metas))
    inlier_file = open(os.path.join(datapath, scene, f"gt_stat.txt"), "w")
    for mask, depth, nocs, meta in bar:
        frame_id = os.path.basename(mask)[:4]
        bar.write(f"processing frame {frame_id}")

        depth = cv2.imread(depth, cv2.IMREAD_ANYDEPTH) / 1000.0
        depth = torch.as_tensor(depth)
        mask = cv2.imread(mask)[:, :, 2]
        mask = torch.as_tensor(mask)
        mask[depth < 1e-5] = 255
        nocs = cv2.imread(nocs, -1)
        nocs = nocs[:, :, [2, 1, 0]]
        nocs = nocs.astype(np.float32) / 255.0
        nocs[:, :, 2] = 1 - nocs[:, :, 2]
        nocs -= 0.5
        nocs = torch.as_tensor(nocs).permute(2, 0, 1)

        with open(meta, "r") as f:
            # [inst_id(1-indexed), cat_id(0-indexed), sym_name)
            meta = [line.strip().split(" ") for line in f.readlines()]
        obj_ids = mask.unique()[:-1].tolist()
        bar.write(f"unique_id = {obj_ids}")

        frame_gt = dict()
        for obj_id in obj_ids:
            obj_mask = mask
            obj_mask = obj_mask == obj_id
            if obj_mask.sum() < 100:
                bar.write("WARNING: no enough points")
                continue
            obj_nocs = nocs
            obj_nocs = obj_nocs.masked_select(obj_mask).view([3, -1])
            obj_depth = depth.masked_select(obj_mask).view([1, -1])
            obj_grid = torch.cartesian_prod(
                torch.arange(obj_mask.size(0)), torch.arange(obj_mask.size(1))
            ).to(obj_nocs.device)
            obj_grid = obj_grid.view(obj_mask.size(0), obj_mask.size(1), 2)
            obj_grid = obj_grid.permute(2, 0, 1).masked_select(obj_mask).view(2, 1, -1)
            obj_grid_y, obj_grid_x = obj_grid.unbind(0)
            obj_X = (obj_grid_x - cx) / (fx)
            obj_Y = (obj_grid_y - cy) / (fy)
            obj_I = torch.ones_like(obj_depth)
            obj_xyz = torch.cat([obj_X, obj_Y, obj_I], dim=0)
            obj_xyz *= obj_depth

            obj_nocs = obj_nocs.permute(1, 0).unsqueeze(0)
            obj_xyz = obj_xyz.permute(1, 0).unsqueeze(0)

            robust_tranf, score = gutils.fast_umeyama_ransac(
                obj_nocs.float(), obj_xyz.float(), 200, 0.01
            )
            bar.write(f"inlier ratio = {score/obj_xyz.size(1)}")

            s, T, R = robust_tranf.s, robust_tranf.T, robust_tranf.R
            obj_xyz_est = (s * obj_nocs) @ R + T

            est_pose = torch.eye(4)
            est_pose[:3, :3] = s[0] * (R[0][:3, :3].t())
            est_pose[:3, 3] = T[0]
            est_pose = est_pose.cpu().numpy().astype(np.float64)
            frame_gt[obj_id] = est_pose
            print(frame_id, obj_id, (score / obj_xyz.size(1)).item(), file=inlier_file)

        with open(os.path.join(datapath, f"{frame_id}_gt.remake.pkl"), "wb") as f:
            pickle.dump(frame_gt, f)

    inlier_file.close()
