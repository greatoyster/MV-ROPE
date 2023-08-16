import os
import glob
import pickle
from tqdm import tqdm
import cv2
import numpy as np
import open3d as o3d

# Set the base directory path where the scenes are located
base_directory = "/mnt/wd8t/Datasets/DROID_DATA/nocs/real/test"
scenes = ["scene_1", "scene_2", "scene_3", "scene_4", "scene_5", "scene_6"]
frame_id = "0000"
scene_id = scenes[0]

scene_dir = os.path.join(base_directory, scene_id)

color = os.path.join(scene_dir, f"{frame_id}_color.png")
mask = os.path.join(scene_dir, f"{frame_id}_mask.pred.png")
nocs = os.path.join(scene_dir, f"{frame_id}_coord.pred.png")

color = cv2.imread(color)
color = color[:, :, [2, 1, 0]]
mask = cv2.imread(mask, -1)[:, :, 2]
nocs = cv2.imread(nocs)
nocs = nocs / 255.0
nocs = nocs[:, :, [2, 1, 0]]
nocs[:, :, 2] = 1 - nocs[:, :, 2]
nocs -= 0.5

obj_ids = np.unique(mask).tolist()[:-1]

for obj_id in obj_ids:
    obj_mask = mask == obj_id
    obj_nocs = nocs[obj_mask]
    obj_color = color[obj_mask]
    obj_pc = o3d.geometry.PointCloud()
    obj_pc.points = o3d.utility.Vector3dVector(obj_nocs)
    obj_pc.colors = o3d.utility.Vector3dVector(obj_color / 255)
    o3d.io.write_point_cloud(f"{scene_id}_{frame_id}_inst_{obj_id}.ply", obj_pc)
