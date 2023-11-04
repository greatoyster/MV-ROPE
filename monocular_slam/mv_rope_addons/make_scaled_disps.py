from glob import glob
from lietorch import SE3, Sim3
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.linear_model import HuberRegressor
import os
import cv2
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# fmt: off
sys.path.append('droid_slam')
import geom.utils as gutils
import geom.projective_ops as pops
import geom.pose_averaging as pose_avg
dataset_path = "/home/yangjq/Datasets/NOCS/real/real_test/scene_1"
disps_data = np.load("/home/yangjq/Projects/MV-ROPE/monocular_slam/reconstructions/scene_1/disps.npy")
timestamps_data = np.load("/home/yangjq/Projects/MV-ROPE/monocular_slam/reconstructions/scene_1/tstamps.npy")
depth_scales = [1.0] * len(timestamps_data)
best_score = -99999
gt_paths = sorted(glob(os.path.join(dataset_path, "*_depth.png")))
for i, ts in tqdm(enumerate(timestamps_data)):
    print(disps_data)
    exit()
    print(disps_data[i] > 0)
    pred_depth = np.reciprocal(disps_data[i]>0)
    ht, wd = pred_depth.shape[:2]
    gt_depth = cv2.imread(
        gt_paths[int(ts)], cv2.IMREAD_ANYDEPTH) / 1000
    gt_depth = cv2.resize(gt_depth, (wd, ht),
                            interpolation=cv2.INTER_NEAREST)

    mask = gt_depth > 1e-5

    masked_gt_depth = gt_depth[mask].reshape(-1)
    masked_pred_depth = pred_depth[mask].reshape(-1, 1)
    reg = HuberRegressor(
        epsilon=1.05, max_iter=20000, fit_intercept=False)
    reg.fit(masked_pred_depth, masked_gt_depth)

    score = reg.score(masked_pred_depth, masked_gt_depth)
    if score > best_score:
        best_score = score
        best_model = reg
    depth_scales[i] = reg.coef_.item()
scale = best_model.coef_.item()
#Here we apply the scale to depth predicted
disps_scaled = disps_data*(1.0/scale)
save_path = "/home/yangjq/Projects/MV-ROPE/monocular_slam/reconstructions/scene_1"

np.save(os.path.join(save_path, "scaled_disps.npy"), disps_scaled)
