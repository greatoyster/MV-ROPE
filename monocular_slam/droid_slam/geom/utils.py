import torch
import numpy as np
from lietorch import Sim3
from pytorch3d.transforms import matrix_to_quaternion
from pytorch3d.ops import corresponding_points_alignment, box3d_overlap
import time
import math


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} took {execution_time} seconds to execute.")
        return result

    return wrapper


def sim3_tensor_to_mat(sim3):
    return Sim3(sim3).matrix()


def mat_to_sim3_tensor(mat):
    if isinstance(mat, np.ndarray):
        print("convert gt from numpy array")
        mat = torch.as_tensor(mat).float()

    sim3_tensor = torch.zeros(8, dtype=torch.float, device=mat.device)

    U, s, Vh = torch.linalg.svd(mat[:3, :3])
    rot_mat = U @ Vh
    rot_mat.unsqueeze_(0)
    rot_quat = matrix_to_quaternion(rot_mat)
    rot_quat = rot_quat[:, [1, 2, 3, 0]]

    sim3_tensor[:3] = mat[:3, 3]
    sim3_tensor[3:7] = rot_quat[:]
    sim3_tensor[7] = s[0]
    return sim3_tensor


def Rst_to_mat(R, s, t):
    mat = torch.eye(4, device=R.device)
    mat[:3, :3] = R.reshape(3, 3) * s
    mat[:3, 3] = t.view(-1)
    return mat


def Rst_to_sim3_tensor(R, s, t):
    mat = Rst_to_mat(R, s, t)
    return mat_to_sim3_tensor(mat)


def umeyama_ransac(
    X,
    Y,
    max_iterations: int = 100,
    inlier_threshold: float = 0.05,
    local_optimization: bool = False,
):
    iteration_count = 0

    so_far_best_model = None
    so_far_best_score = 0
    N = X.size(1)

    while iteration_count < max_iterations:
        iteration_count += 1
        minimal_sample_indices = torch.randperm(N)[:4]
        minimal_X = X[:, minimal_sample_indices, :]
        minimal_Y = Y[:, minimal_sample_indices, :]
        transf = corresponding_points_alignment(minimal_X, minimal_Y, None, True)
        s, T, R = transf.s, transf.T, transf.R
        residual = (s * X) @ R + T - Y
        residual = residual.norm(dim=2)
        score = (residual < inlier_threshold).sum()
        if score > so_far_best_score:
            so_far_best_score = score
            so_far_best_model = transf

    s, T, R = so_far_best_model.s, so_far_best_model.T, so_far_best_model.R
    residual = (s * X) @ R + T - Y
    residual = residual.norm(dim=2)
    mask = residual < inlier_threshold
    X_mask = X[mask].unsqueeze(0)
    Y_mask = Y[mask].unsqueeze(0)
    so_far_best_model = corresponding_points_alignment(X_mask, Y_mask, None, True)

    return so_far_best_model, so_far_best_score


def fast_umeyama_ransac(
    X,
    Y,
    max_iterations: int = 100,
    inlier_threshold: float = 0.05,
    local_optimization: bool = False,
):
    so_far_best_model = None
    so_far_best_score = 0
    N = X.size(1)
    assert N >= 4

    selected_sample_Xs = []
    selected_sample_Ys = []

    for i in range(max_iterations):
        minimal_sample_indices = torch.randperm(N)[:4]
        selected_sample_Xs.append(X[:, minimal_sample_indices, :])
        selected_sample_Ys.append(Y[:, minimal_sample_indices, :])

    selected_sample_Xs = torch.cat(selected_sample_Xs, dim=0)
    selected_sample_Ys = torch.cat(selected_sample_Ys, dim=0)

    R, T, s = corresponding_points_alignment(
        selected_sample_Xs, selected_sample_Ys, None, True
    )

    X_repeat = X.expand(max_iterations, -1, -1)
    Y_repeat = Y.expand(max_iterations, -1, -1)

    residual = s[:, None, None] * torch.bmm(X_repeat, R) + T[:, None, :] - Y_repeat

    residual = residual.norm(dim=-1)
    score = residual < inlier_threshold
    score = score.sum(dim=-1)
    so_far_best_index = torch.argmax(score)
    so_far_best_score = score[so_far_best_index]

    R = R[so_far_best_index].unsqueeze(0)
    T = T[so_far_best_index].unsqueeze(0)
    s = s[so_far_best_index].unsqueeze(0)

    residual = (s * X) @ R + T - Y
    residual = residual.norm(dim=2)
    mask = residual < inlier_threshold
    X_mask = X[mask].unsqueeze(0)
    Y_mask = Y[mask].unsqueeze(0)
    so_far_best_model = corresponding_points_alignment(X_mask, Y_mask, None, True)

    return so_far_best_model, so_far_best_score


def fast_umeyama_ransac_with_shape_optimization(
    obj_nocs,
    obj_xyz,
    thres1=0.01,
    thres2_ratio=0.65,
    step_size=0.05,
    refine_iterations=10,
    ransac_iteration=400,
):
    robust_tranf, score = fast_umeyama_ransac(
        obj_nocs, obj_xyz, ransac_iteration, thres1
    )
    s, T, R = robust_tranf.s, robust_tranf.T, robust_tranf.R
    for i in range(refine_iterations):
        robust_tranf, score = fast_umeyama_ransac(
            obj_nocs, obj_xyz, ransac_iteration, thres1
        )
        s, T, R = robust_tranf.s, robust_tranf.T, robust_tranf.R

        obj_xyz_est = (s * obj_nocs) @ R + T
        residual = obj_xyz - obj_xyz_est
        residual = residual.norm(dim=2)
        # thres2_index = int(residual.size(1) * thres2_ratio)
        # sorted_values, indices = torch.sort(residual)

        # thres2 = sorted_values[0][thres2_index]
        thres2 = thres2_ratio
        mask2 = torch.logical_and(thres1 <= residual, residual < thres2)
        obj_nocs_copy = obj_nocs.clone()
        M_nocs_points = obj_nocs_copy[mask2].unsqueeze(0)
        M_xyz_points = obj_xyz[mask2].unsqueeze(0)
        M_xyz_points_transformed = (M_xyz_points - T) @ R.transpose(-1, -2) / s
        M_nocs_points_centroid = M_nocs_points.mean(dim=1, keepdims=True)
        M_nocs_points += step_size * (M_xyz_points_transformed - M_nocs_points)
        offset = M_nocs_points.mean(dim=1, keepdims=True) - M_nocs_points_centroid
        M_nocs_points -= offset
        obj_nocs[mask2] = M_nocs_points.squeeze(0)

    robust_tranf, score = fast_umeyama_ransac(
        obj_nocs, obj_xyz, ransac_iteration, thres1
    )
    return robust_tranf, score


def iou3d(sim3_mat_a, sim3_mat_b):
    "return: intersection volume, iou"
    device = sim3_mat_a.device
    box_corner_vertices = torch.as_tensor(
        [
            [0, 0, 0, 1.0],
            [1, 0, 0, 1.0],
            [1, 1, 0, 1.0],
            [0, 1, 0, 1.0],
            [0, 0, 1, 1.0],
            [1, 0, 1, 1.0],
            [1, 1, 1, 1.0],
            [0, 1, 1, 1.0],
        ],
        device=device,
    ).transpose(0, 1)
    box_a = sim3_mat_a @ box_corner_vertices
    box_b = sim3_mat_b @ box_corner_vertices
    box_a = box_a.transpose(0, 1)[:, :3].unsqueeze(0)
    box_b = box_b.transpose(0, 1)[:, :3].unsqueeze(0)
    return box3d_overlap(box_a, box_b)


def pose_diff(sim3_mat_a, sim3_mat_b):
    # return rotation difference in degree and translation in meters
    Ua, sa, Vha = torch.linalg.svd(sim3_mat_a[:3, :3])
    Ub, sb, Vhb = torch.linalg.svd(sim3_mat_b[:3, :3])
    Ra = Ua @ Vha
    Rb = Ub @ Vhb
    diff = Ra @ Rb.t()
    diff = matrix_to_quaternion(diff.unsqueeze(0))
    iden = matrix_to_quaternion(torch.eye(3, device=diff.device).unsqueeze(0))
    cosine = torch.sum(diff[0] * iden[0])
    cosine = torch.arccos(cosine) * (360 / (2 * torch.pi))
    t = torch.linalg.norm(sim3_mat_a[:3, 3] - sim3_mat_b[:3, 3])
    return cosine, t


def symmetric_iou3d(sim3_mat_a, sim3_mat_b, steps=360):
    device = sim3_mat_a.device
    max_vol = 0
    max_iou = 0
    for i in range(steps):
        theta = i * 2 * torch.pi / steps
        R = torch.as_tensor(
            [
                [np.cos(theta), 0, np.sin(theta), 0],
                [0, 1, 0, 0],
                [-np.sin(theta), 0, np.cos(theta), 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float,
            device=device,
        )

        vol, iou = iou3d(sim3_mat_a @ R, sim3_mat_b)
        if iou > max_iou:
            max_iou = iou
            max_vol = vol
    return (max_vol, max_iou)


def symmetric_pose_diff(sim3_mat_a, sim3_mat_b, steps=360):
    device = sim3_mat_a.device
    min_rot = 10000
    min_trans = 10000
    for i in range(steps):
        theta = i * 2 * torch.pi / steps
        R = torch.as_tensor(
            [
                [np.cos(theta), 0, np.sin(theta), 0],
                [0, 1, 0, 0],
                [-np.sin(theta), 0, np.cos(theta), 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float,
            device=device,
        )

        rot, trans = pose_diff(sim3_mat_a @ R, sim3_mat_b)
        if rot < min_rot:
            min_rot = rot
            min_trans = trans
    return (min_rot, min_trans)


def fast_symmetric_iou3d(sim3_mat_a, sim3_mat_b, steps=360):
    device = sim3_mat_a.device

    thetas = torch.as_tensor(
        [i * 2 * torch.pi / steps for i in range(steps)],
        dtype=torch.float,
        device=device,
    )
    Rs = torch.eye(4, dtype=torch.float, device=device)[None, ::].repeat(steps, 1, 1)
    Rs[:, 0, 0] = torch.cos(thetas)
    Rs[:, 2, 2] = torch.cos(thetas)
    Rs[:, 0, 2] = torch.sin(thetas)
    Rs[:, 2, 0] = -torch.sin(thetas)

    sim3_mat_a_batch = torch.bmm(sim3_mat_a[None, ::].expand_as(Rs), Rs)
    sim3_mat_b_batch = sim3_mat_b[None, ::].expand_as(Rs)
    box_corner_vertices = (
        torch.as_tensor(
            [
                [
                    [0, 0, 0, 1.0],
                    [1, 0, 0, 1.0],
                    [1, 1, 0, 1.0],
                    [0, 1, 0, 1.0],
                    [0, 0, 1, 1.0],
                    [1, 0, 1, 1.0],
                    [1, 1, 1, 1.0],
                    [0, 1, 1, 1.0],
                ]
            ],
            device=device,
        )
        .transpose(-2, -1)
        .expand(steps, -1, -1)
    )

    boxes_a = torch.bmm(sim3_mat_a_batch, box_corner_vertices).transpose(-2, -1)[
        :, :, :3
    ]
    boxes_b = torch.bmm(sim3_mat_b_batch, box_corner_vertices).transpose(-2, -1)[
        :, :, :3
    ]

    vols, ious = box3d_overlap(boxes_a, boxes_b)
    max_index = torch.argmax(torch.diagonal(ious))
    max_vol = vols[max_index, max_index]
    max_iou = ious[max_index, max_index]

    return (max_vol, max_iou)


def fast_symmetric_pose_diff(sim3_mat_a, sim3_mat_b, steps=360):
    device = sim3_mat_a.device

    t = torch.linalg.norm(sim3_mat_a[:3, 3] - sim3_mat_b[:3, 3])

    thetas = torch.as_tensor(
        [i * 2 * torch.pi / steps for i in range(steps)],
        dtype=torch.float,
        device=device,
    )
    Rs = torch.eye(4, dtype=torch.float, device=device)[None, ::].repeat(steps, 1, 1)
    Rs[:, 0, 0] = torch.cos(thetas)
    Rs[:, 2, 2] = torch.cos(thetas)
    Rs[:, 0, 2] = torch.sin(thetas)
    Rs[:, 2, 0] = -torch.sin(thetas)

    sim3_mat_a_batch = torch.bmm(sim3_mat_a[None, ::].expand_as(Rs), Rs)
    sim3_mat_b_batch = sim3_mat_b[None, ::].expand_as(Rs)

    Ua, sa, Vha = torch.linalg.svd(sim3_mat_a_batch[:, :3, :3])
    Ub, sb, Vhb = torch.linalg.svd(sim3_mat_b_batch[:, :3, :3])

    Ra = torch.bmm(Ua, Vha)
    Rb = torch.bmm(Ub, Vhb)

    diff = torch.bmm(Ra, Rb.transpose(-1, -2))
    diff = matrix_to_quaternion(diff)
    iden = matrix_to_quaternion(
        torch.eye(3, device=diff.device).unsqueeze(0)
    ).expand_as(diff)
    cosine = torch.sum(diff * iden, dim=-1)
    cosine = torch.arccos(cosine)
    cosine = cosine.min()
    cosine = cosine * (360 / (2 * torch.pi))
    return (cosine, t)


def compute_RT_errors(sRT_1, sRT_2, symmetry=False, handle_visibility=True):
    """
    Args:
        sRT_1: [4, 4]. homogeneous affine transformation
        sRT_2: [4, 4]. homogeneous affine transformation

    Returns:
        theta: angle difference of R in degree
        shift: l2 difference of T in centimeter
    """
    sRT_1 = sRT_1.cpu().numpy()
    sRT_2 = sRT_2.cpu().numpy()
    # make sure the last row is [0, 0, 0, 1]
    if sRT_1 is None or sRT_2 is None:
        return -1
    try:
        assert np.array_equal(sRT_1[3, :], sRT_2[3, :])
        assert np.array_equal(sRT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(sRT_1[3, :], sRT_2[3, :])
        exit()

    R1 = sRT_1[:3, :3] / np.cbrt(np.linalg.det(sRT_1[:3, :3]))
    T1 = sRT_1[:3, 3]
    R2 = sRT_2[:3, :3] / np.cbrt(np.linalg.det(sRT_2[:3, :3]))
    T2 = sRT_2[:3, 3]
    # symmetric when rotating around y-axis
    if symmetry and handle_visibility:
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        cos_theta = y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2))
    else:
        R = R1 @ R2.transpose()
        cos_theta = (np.trace(R) - 1) / 2

    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
    shift = np.linalg.norm(T1 - T2)

    return theta.item(), shift.item()


def asymmetric_3d_iou(bbox_3d_1, bbox_3d_2):
    bbox_1_max = torch.max(bbox_3d_1, dim=0)[0]
    bbox_1_min = torch.min(bbox_3d_1, dim=0)[0]
    bbox_2_max = torch.max(bbox_3d_2, dim=0)[0]
    bbox_2_min = torch.min(bbox_3d_2, dim=0)[0]

    overlap_min = torch.max(bbox_1_min, bbox_2_min)
    overlap_max = torch.min(bbox_1_max, bbox_2_max)

    # intersections and union
    overlap_dims = overlap_max - overlap_min
    if torch.any(overlap_dims < 0):
        intersections = torch.tensor(0.0)
    else:
        intersections = torch.prod(overlap_dims)

    union = (
        torch.prod(bbox_1_max - bbox_1_min)
        + torch.prod(bbox_2_max - bbox_2_min)
        - intersections
    )

    overlaps = intersections / union
    return overlaps.item()


def compute_3d_IoU(sRT_1, sRT_2, symmetry=False, handle_visibility=True):
    """Computes IoU overlaps between two 3D bboxes."""
    device = sRT_1.device

    if symmetry and handle_visibility:
        steps = 360
        thetas = torch.as_tensor(
            [i * 2 * torch.pi / steps for i in range(steps)],
            dtype=torch.float,
            device=device,
        )
        Rs = torch.eye(4, dtype=torch.float, device=device)[None, ::].repeat(
            steps, 1, 1
        )
        Rs[:, 0, 0] = torch.cos(thetas)
        Rs[:, 2, 2] = torch.cos(thetas)
        Rs[:, 0, 2] = torch.sin(thetas)
        Rs[:, 2, 0] = -torch.sin(thetas)

        sim3_mat_a_batch = torch.bmm(sRT_1[None, ::].expand_as(Rs), Rs)
        sim3_mat_b_batch = sRT_2[None, ::].expand_as(Rs)
        box_corner_vertices = (
            torch.as_tensor(
                [
                    [
                        [0, 0, 0, 1.0],
                        [1, 0, 0, 1.0],
                        [1, 1, 0, 1.0],
                        [0, 1, 0, 1.0],
                        [0, 0, 1, 1.0],
                        [1, 0, 1, 1.0],
                        [1, 1, 1, 1.0],
                        [0, 1, 1, 1.0],
                    ]
                ],
                device=device,
            )
            .transpose(-2, -1)
            .expand(steps, -1, -1)
        )

        boxes_a = torch.bmm(sim3_mat_a_batch, box_corner_vertices).transpose(-2, -1)[
            :, :, :3
        ]
        boxes_b = torch.bmm(sim3_mat_b_batch, box_corner_vertices).transpose(-2, -1)[
            :, :, :3
        ]
        max_iou = 0
        for i in range(steps):
            cur_iou = asymmetric_3d_iou(boxes_a[i], boxes_b[i])
            max_iou = max(cur_iou, max_iou)
        return max_iou
    else:
        box_corner_vertices = torch.as_tensor(
            [
                [0, 0, 0, 1.0],
                [1, 0, 0, 1.0],
                [1, 1, 0, 1.0],
                [0, 1, 0, 1.0],
                [0, 0, 1, 1.0],
                [1, 0, 1, 1.0],
                [1, 1, 1, 1.0],
                [0, 1, 1, 1.0],
            ],
            device=device,
        ).transpose(0, 1)
        box_a = sRT_1 @ box_corner_vertices
        box_b = sRT_2 @ box_corner_vertices
        box_a = box_a.transpose(0, 1)[:, :3]
        box_b = box_b.transpose(0, 1)[:, :3]
        return asymmetric_3d_iou(box_a, box_b)
