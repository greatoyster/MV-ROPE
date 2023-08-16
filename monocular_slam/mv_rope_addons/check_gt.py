"""
check whether or not there is invalid ground truth poses and hand_visibility in original nocs label
"""

import glob
import pickle
from tqdm import tqdm
import os
import cv2
import numpy as np

# Set the base directory path where the scenes are located
base_directory = "/mnt/wd8t/Datasets/DROID_DATA/nocs/real/real_test/"
scenes = ["scene_1", "scene_2", "scene_3", "scene_4", "scene_5", "scene_6"]

# # Iterate over scenes
# for scene in scenes:
#     # Set the directory path for the current scene
#     directory = base_directory + scene

#     # Glob all the results*.pkl files
#     file_list = glob.glob(directory + '/results**')

#     # Load and print the keys of each file
#     for file_path in tqdm(file_list, desc=f'Scene {scene}'):
#         with open(file_path, 'rb') as file:
#             data = pickle.load(file)
#             if len(list(data.keys())) > 2:
#                 print("Keys:", list(data.keys()))


scenes = ["scene_22", "scene_21" ,"scene_23", "scene_24","scene_26", "scene_25", "scene_1", "scene_2", "scene_3", "scene_4", "scene_5", "scene_6" , "scene_7", "scene_8", "scene_9"]
for scene in scenes:
    # Iterate through all files in the folder
    folder_path = base_directory + scene
    for filename in os.listdir(folder_path):
        if filename.endswith("mask.pred.png"):
            # Extract the unique number from the filename
            mask = cv2.imread(os.path.join(folder_path, filename))[:, :, 2]
            print(scene, filename, np.unique(mask))
