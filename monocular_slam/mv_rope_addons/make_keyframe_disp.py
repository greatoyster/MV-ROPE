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
# fmt: on


"""
results.txt
scene_id(i), obj_id(i), cat_id(i), inlier_ratio(f), iou(f), trans_error(f), rot_error(f) 
"""


export_scaled_depth_error = True

use_global_scale = True  # True: used in monocular video; False: used in RGBD video

scenes = ["scene_1"]
# Initialize an empty dictionary
data_dict = {}
sym_cat = [1, 2, 4]

evaluation_statistics = dict(
    pose_count=0,
    deg10_trans10=0,
    deg10_trans5=0,
    deg5_trans10=0,
    deg5_trans5=0,
    iou25=0,
    iou50=0,
    iou75=0,
)

# Specify the file names and corresponding keys in the dictionary
file_names = {
    "disps.npy": "disps",
    "images.npy": "images",
    "intrinsics.npy": "intrinsics",
    "poses.npy": "poses",
    "tstamps.npy": "timestamps",
    "nocs.npy": "nocs",
    "masks.npy": "masks",
    "gt_obj_poses.npy": "gt_obj_poses",
    "obj_poses.npy": "obj_poses",
    "active_objs.npy": "active_objs",
    "obj_pose_scores.npy": "obj_pose_scores",
}

what_we_want = [18, 108, 251]

for scene in scenes:
    datapath = os.path.join(
        "/mnt/wd8t/Datasets/DROID_DATA/nocs/real/real_test/", scene)
    # reconstruction_path = os.path.join("reconstructions", scene) # "reconstructions/scene_1_fw_4_kt_2.5_zd"
    exp_variants = glob(f"reconstructions/{scene}_fw_16_kt_3.5*")
    print(exp_variants)
    
    if use_global_scale:
        exp_variants = [i for i in exp_variants if i.endswith("zd")]
    else:
        exp_variants = [i for i in exp_variants if not i.endswith("zd")]
    print(exp_variants)
    for e in exp_variants:
        print(e, "---------------------------")
        reconstruction_path = e
        metrics = []
        gts = sorted(glob(os.path.join(datapath, "*.remake.pkl")))
        gt_depth_paths = sorted(glob(os.path.join(datapath, "*depth.png")))

        for file_name, key in file_names.items():
            file_path = os.path.join(reconstruction_path, file_name)
            data_dict[key] = np.load(file_path)

        local_window = (0, None, 1)
        start, end, stride = local_window

        # Access the loaded data
        disps_data = torch.as_tensor(data_dict["disps"][start:end:stride])
        images_data = torch.as_tensor(data_dict["images"][start:end:stride])
        intrinsics_data = torch.as_tensor(
            data_dict["intrinsics"][start:end:stride] * 8.0)
        poses_data = torch.as_tensor(data_dict["poses"][start:end:stride])
        timestamps_data = torch.as_tensor(
            data_dict["timestamps"][start:end:stride])
        nocs_data = torch.as_tensor(data_dict["nocs"][start:end:stride])
        masks_data = torch.as_tensor(data_dict["masks"][start:end:stride])
        gt_obj_poses_data = torch.as_tensor(
            data_dict["gt_obj_poses"][start:end:stride])
        obj_poses_data = torch.as_tensor(
            data_dict["obj_poses"][start:end:stride])
        active_objs_data = torch.as_tensor(
            data_dict["active_objs"][start:end:stride])
        obj_pose_scores_data = torch.as_tensor(
            data_dict["obj_pose_scores"][start:end:stride]
        )

        timestamps_data = timestamps_data.long().tolist()
        print(timestamps_data)
        best_score = -100000
        best_model = None
        scale = 1.0
        depth_scales = [1.0] * len(timestamps_data)
        if use_global_scale:
            for i, ts in tqdm(enumerate(timestamps_data)):
                pred_depth = 1 / disps_data[i].cpu().numpy()
                ht, wd = pred_depth.shape[:2]

                gt_depth = cv2.imread(
                    gt_depth_paths[ts], cv2.IMREAD_ANYDEPTH) / 1000
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

        if export_scaled_depth_error:
            for i, ts in tqdm(enumerate(timestamps_data)):
                if not ts in what_we_want:
                    continue
                pred_depth = 1 / disps_data[i].cpu().numpy()
                ht, wd = pred_depth.shape[:2]

                gt_depth = cv2.imread(
                    gt_depth_paths[ts], cv2.IMREAD_ANYDEPTH) / 1000
                gt_depth = cv2.resize(gt_depth, (wd, ht),
                                      interpolation=cv2.INTER_NEAREST)

                mask = gt_depth > 1e-5

                masked_gt_depth = gt_depth[mask].reshape(-1)
                masked_pred_depth = pred_depth[mask].reshape(-1, 1)

                score = reg.score(masked_pred_depth, masked_gt_depth)
                vmax = max(np.max(pred_depth),np.max(gt_depth))
                vmin = min(np.min(pred_depth),np.min(gt_depth))

                pred_depth = pred_depth* best_model.coef_
                corrected_depth = pred_depth
                fig = plt.figure()
                plt.imshow(corrected_depth,vmin=vmin,vmax=vmax)
                plt.axis("off")
                plt.savefig(f"{scene}_{ts}_depth_pred.png", dpi=500, bbox_inches='tight')
                
                
                fig = plt.figure()
                plt.imshow(gt_depth,vmin=vmin,vmax=vmax)
                plt.axis("off")
                plt.savefig(f"{scene}_{ts}_depth_gt.png", dpi=500, bbox_inches='tight')

                # plt.imshow(gt_depth)
                # plt.savefig(
                #     f"{scene}_{str(ts).zfill(4)}_gt_and_corrected.b.png")
