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
scenes = [ "scene_31"]
for scene in scenes:
    metas = sorted(glob(os.path.join(datapath, scene, "*_meta.pred.txt")))
    bar = tqdm(metas)
    for meta in bar:
        frame_id = os.path.basename(meta)[:4]
        bar.write(f"processing frame {frame_id}")
        frame_gt = dict()
        with open(
            os.path.join(datapath, scene, f"{frame_id}_gt.remake.pkl"), "wb"
        ) as f:
            pickle.dump(frame_gt, f)
