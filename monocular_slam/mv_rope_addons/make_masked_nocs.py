import os
import glob
import pickle
from tqdm import tqdm
import cv2
import numpy as np

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
mask = cv2.imread(mask, -1)[:, :, 2]
mask = mask != 255
nocs = cv2.imread(nocs)
color[mask] = nocs[mask]

cv2.imwrite(f"{scene_id}_{frame_id}_masked_nocs.png", color)
