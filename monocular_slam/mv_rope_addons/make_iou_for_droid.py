import glob
import os
import sys
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# fmt: off
sys.path.append('droid_slam')
import geom.utils as gutils


dataset_path = "/home/yangjq/Datasets/dyna_rope_data/droid/scene_1"
pkl_files = sorted(glob.glob(os.path.join(dataset_path, "*.pkl")))
for pkl_file in pkl_files:
    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)
    print(data)
    exit()
