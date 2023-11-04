import numpy as np
import torch
from lietorch import Sim3, SE3
from glob import glob
import os
import cv2


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


def draw_vertice_and_lines(
    raw_image, vertice, edges, intrinsics, vert_color, line_color
):
    # draw points
    fx, fy, cx, cy = intrinsics
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


def draw_3dbox(image_path, RTs, intrinsics):
    RTs = torch.from_numpy(RTs)
    image = cv2.imread(image_path)
    new_vertice = vertices.clone()
    RTs = RTs.float()
    new_vertice = RTs[:3, :3] @ new_vertice.t() + RTs[:3, 3:4]

    new_vertice *= 100
    new_vertice = new_vertice.t()
    print(new_vertice, RTs)
    image_new = draw_vertice_and_lines(image, new_vertice, edges, intrinsics, 4, 100)
    base_path, extension = image_path.rsplit(".", 1)

    # Add the new text before the extension
    new_file_path = base_path + "3dbox" + "." + extension
    cv2.imwrite(new_file_path, image_new)
    return


intrinsics = [591.0125, 590.16775, 322.525, 244.11084]
import pickle

# Specify the path to the .pkl file
file_path = "/home/yangjq/Projects/IST-Net/log/ist_net_default/eval_epoch30/results_test_scene_1_0000.pkl"

# Load the data from the .pkl file
with open(file_path, "rb") as file:
    data = pickle.load(file)

# Now you can use the loaded data
# print(data)

RTs = data["gt_RTs"][0]
print(RTs)
# Now the task have been completed, reminding that RTs is a numpy and intrinsics is a 1x4 matrix
draw_3dbox(
    "/home/yangjq/Projects/IST-Net/data/real/real_test/scene_1/0000_color.png",
    RTs,
    intrinsics,
)
