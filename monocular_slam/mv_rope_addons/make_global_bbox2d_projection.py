"""
project object pose unit cube in 2d images    
"""

import numpy as np
import torch
from lietorch import Sim3, SE3
from glob import glob
import os
import cv2


def generate_color(time):
    # Adjust the color properties based on time
    frequency = 0.8  # Controls the speed of color change
    phase_shift = 0  # Controls the starting point of the color change
    amplitude = 127  # Controls the range of color variation

    # Generate the RGB values using sine waves
    red = int(
        amplitude * np.sin(frequency * time + 2 * np.pi * 0 / 3 + phase_shift) + 128
    )
    green = int(
        amplitude * np.sin(frequency * time + 2 * np.pi * 1 / 3 + phase_shift) + 128
    )
    blue = int(
        amplitude * np.sin(frequency * time + 2 * np.pi * 2 / 3 + phase_shift) + 128
    )

    return (red, green, blue)


vertices = torch.tensor(
    [
        [0, 0, 0],  # vertex 1
        [1, 0, 0],  # vertex 2
        [0, 1, 0],  # vertex 3
        [1, 1, 0],  # vertex 4
        [0, 0, 1],  # vertex 5
        [1, 0, 1],  # vertex 6
        [0, 1, 1],  # vertex 7
        [1, 1, 1],  # vertex 8
    ],
    dtype=torch.float32,
)


vertices -= 0.5
# vertices *= 0.55

edges = (
    torch.tensor(
        [
            [0, 1],  # edge 1
            [1, 3],  # edge 2
            [3, 2],  # edge 3
            [2, 0],  # edge 4
            [4, 5],  # edge 5
            [5, 7],  # edge 6
            [7, 6],  # edge 7
            [6, 4],  # edge 8
            [0, 4],  # edge 9
            [1, 5],  # edge 10
            [2, 6],  # edge 11
            [3, 7],  # edge 12
        ],
        dtype=torch.int64,
    )
    .cpu()
    .numpy()
)

# lineset = vertices[edges]

# print(lineset)


def process_data(index):
    image = images_data[index]
    image = torch.as_tensor(image)

    nocs = nocs_data[index]
    nocs = torch.as_tensor(nocs)

    disp = disps_data[index]
    disp = torch.as_tensor(disp)

    mask = masks_data[index]
    mask = torch.as_tensor(mask)

    pose = poses_data[index]
    pose = torch.as_tensor(pose)

    return (image, nocs, disp, mask, pose)


def draw_vertice_and_lines(
    raw_image, vertice, edges, intrinsics, vert_color, line_color
):
    # draw points
    fx, fy, cy, cy = intrinsics
    x = vertice[:, 0]
    y = vertice[:, 1]
    z = vertice[:, 2]

    # Apply perspective projection equations
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy

    # Create the 2D projection tensor
    points_2d = torch.stack((u, v), dim=1).cpu().numpy().astype(np.int32)
    for edge in edges:
        raw_image = cv2.line(
            raw_image,
            points_2d[edge[0]],
            points_2d[edge[1]],
            line_color,
            1,
            cv2.LINE_AA,
        )
    for point in points_2d:
        raw_image = cv2.circle(raw_image, point, 3, vert_color, -1, cv2.LINE_AA)
    # draw lines

    return raw_image


# load image, intrinsics and object poses

# Initialize an empty dictionary
data_dict = {}

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
    "global_obj_poses.npy": "global_obj_poses",
    "box_scales.npy": "box_scales"

}

scenes = [ "scene_1"] #, "scene_35", "scene_21"]

for scene in scenes:
    datapath = os.path.join("/mnt/wd8t/Datasets/DROID_DATA/nocs/real/real_test/", scene)
    reconstruction_path = os.path.join("reconstructions", scene)
    gts = sorted(glob(os.path.join(datapath, "*.remake.pkl")))

    with open(os.path.join(datapath, "0001_meta.pred.txt"), "r") as f:
        # [inst_id(1-indexed), cat_id(0-indexed), sym_name)
        meta = [line.strip().split(" ") for line in f.readlines()]

    print(meta)
    meta = {int(m[0]): int(m[1]) for m in meta if len(m) == 3}
    print(meta)

    for file_name, key in file_names.items():
        file_path = os.path.join(reconstruction_path, file_name)
        data_dict[key] = np.load(file_path)


    # Access the loaded data
    disps_data = torch.as_tensor(data_dict["disps"])
    images_data = torch.as_tensor(data_dict["images"])
    intrinsics_data = torch.as_tensor(data_dict["intrinsics"] * 8.0)
    poses_data = torch.as_tensor(data_dict["poses"])  # world to camera
    timestamps_data = torch.as_tensor(data_dict["timestamps"])
    nocs_data = torch.as_tensor(data_dict["nocs"])
    masks_data = torch.as_tensor(data_dict["masks"])
    gt_obj_poses_data = torch.as_tensor(data_dict["gt_obj_poses"])
    obj_poses_data = torch.as_tensor(data_dict["obj_poses"])
    active_objs_data = torch.as_tensor(data_dict["active_objs"])
    obj_pose_scores_data = torch.as_tensor(data_dict["obj_pose_scores"])
    global_obj_poses_data = torch.as_tensor(data_dict["global_obj_poses"])
    box_scales_data = torch.as_tensor(data_dict["box_scales"]) * 2
    
    ref_idx = 0

    for ref_idx in range(len(images_data)):
        
        image_ref = images_data[ref_idx].cpu().permute(1, 2, 0).contiguous().numpy()
        world_to_cam = SE3(poses_data[ref_idx]).matrix()

        fx, fy, cx, cy = (
            intrinsics_data[0, 0],
            intrinsics_data[0, 1],
            intrinsics_data[0, 2],
            intrinsics_data[0, 3],
        )

        for i in range(global_obj_poses_data.size(0)):
            if global_obj_poses_data[i].sum() < 1e-5:
                print(f"no object id {i}")
                continue

            nocs_to_world = Sim3(global_obj_poses_data[i]).matrix()
            nocs_to_cam = world_to_cam @ nocs_to_world
            new_vertice = vertices.clone()
            print("box_scale",box_scales_data[i])
            new_vertice[:, 0] *= box_scales_data[i,0]
            new_vertice[:, 1] *= box_scales_data[i,1]
            new_vertice[:, 2] *= box_scales_data[i,2]

            new_vertice = nocs_to_cam[:3, :3] @ new_vertice.t() + nocs_to_cam[:3, 3:4]
            new_vertice = new_vertice.t()       

            vert_color = generate_color(2 * i - 1)
            line_color = generate_color(2 * i)
            image_ref = draw_vertice_and_lines(
                image_ref, new_vertice, edges, (fx, fy, cx, cy), vert_color, line_color
            )
        cv2.imwrite(f"2d_pro/{scene}_{int(timestamps_data[ref_idx].item())}_box_projection.png", image_ref)
    

