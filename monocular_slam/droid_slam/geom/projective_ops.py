import torch
import torch.nn.functional as F
import lietorch
from lietorch import SE3, Sim3
import ipdb as pdb

MIN_DEPTH = 0.2


def extract_intrinsics(intrinsics):
    return intrinsics[..., None, None, :].unbind(dim=-1)


def coords_grid(ht, wd, **kwargs):
    y, x = torch.meshgrid(
        torch.arange(ht).to(**kwargs).float(), torch.arange(wd).to(**kwargs).float()
    )

    return torch.stack([x, y], dim=-1)


def iproj(disps, intrinsics, jacobian=False):
    """pinhole camera inverse projection"""
    ht, wd = disps.shape[2:]
    fx, fy, cx, cy = extract_intrinsics(intrinsics)

    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float(),
    )

    i = torch.ones_like(disps)
    X = (x - cx) / fx
    Y = (y - cy) / fy
    pts = torch.stack([X, Y, i, disps], dim=-1)

    if jacobian:
        J = torch.zeros_like(pts)
        J[..., -1] = 1.0
        return pts, J

    return pts, None


def proj(Xs, intrinsics, jacobian=False, return_depth=False):
    """pinhole camera projection"""
    fx, fy, cx, cy = extract_intrinsics(intrinsics)
    X, Y, Z, D = Xs.unbind(dim=-1)

    Z = torch.where(Z < 0.5 * MIN_DEPTH, torch.ones_like(Z), Z)
    d = 1.0 / Z

    x = fx * (X * d) + cx
    y = fy * (Y * d) + cy
    if return_depth:
        coords = torch.stack([x, y, D * d], dim=-1)
    else:
        coords = torch.stack([x, y], dim=-1)

    if jacobian:
        B, N, H, W = d.shape
        o = torch.zeros_like(d)
        proj_jac = torch.stack(
            [
                fx * d,
                o,
                -fx * X * d * d,
                o,
                o,
                fy * d,
                -fy * Y * d * d,
                o,
                # o,     o,    -D*d*d,  d,
            ],
            dim=-1,
        ).view(B, N, H, W, 2, 4)

        return coords, proj_jac

    return coords, None


def actp(Gij, X0, jacobian=False):
    """action on point cloud"""
    X1 = Gij[:, :, None, None] * X0

    if jacobian:
        X, Y, Z, d = X1.unbind(dim=-1)
        o = torch.zeros_like(d)
        B, N, H, W = d.shape

        if isinstance(Gij, SE3):
            Ja = torch.stack(
                [
                    d,
                    o,
                    o,
                    o,
                    Z,
                    -Y,
                    o,
                    d,
                    o,
                    -Z,
                    o,
                    X,
                    o,
                    o,
                    d,
                    Y,
                    -X,
                    o,
                    o,
                    o,
                    o,
                    o,
                    o,
                    o,
                ],
                dim=-1,
            ).view(B, N, H, W, 4, 6)

        elif isinstance(Gij, Sim3):
            Ja = torch.stack(
                [
                    d,
                    o,
                    o,
                    o,
                    Z,
                    -Y,
                    X,
                    o,
                    d,
                    o,
                    -Z,
                    o,
                    X,
                    Y,
                    o,
                    o,
                    d,
                    Y,
                    -X,
                    o,
                    Z,
                    o,
                    o,
                    o,
                    o,
                    o,
                    o,
                    o,
                ],
                dim=-1,
            ).view(B, N, H, W, 4, 7)

        return X1, Ja

    return X1, None


def projective_transform(
    poses, depths, intrinsics, ii, jj, jacobian=False, return_depth=False
):
    """map points from ii->jj"""

    # inverse project (pinhole)
    X0, Jz = iproj(depths[:, ii], intrinsics[:, ii], jacobian=jacobian)

    # transform
    Gij = poses[:, jj] * poses[:, ii].inv()

    Gij.data[:, ii == jj] = torch.as_tensor(
        [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device="cuda"
    )
    X1, Ja = actp(Gij, X0, jacobian=jacobian)

    # project (pinhole)
    x1, Jp = proj(X1, intrinsics[:, jj], jacobian=jacobian, return_depth=return_depth)

    # exclude points too close to camera
    valid = ((X1[..., 2] > MIN_DEPTH) & (X0[..., 2] > MIN_DEPTH)).float()
    valid = valid.unsqueeze(-1)

    if jacobian:
        # Ji transforms according to dual adjoint
        Jj = torch.matmul(Jp, Ja)
        Ji = -Gij[:, :, None, None, None].adjT(Jj)

        Jz = Gij[:, :, None, None] * Jz
        Jz = torch.matmul(Jp, Jz.unsqueeze(-1))

        return x1, valid, (Ji, Jj, Jz)

    return x1, valid


def induced_flow(poses, disps, intrinsics, ii, jj):
    """optical flow induced by camera motion"""

    ht, wd = disps.shape[2:]
    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float(),
    )

    coords0 = torch.stack([x, y], dim=-1)
    coords1, valid = projective_transform(poses, disps, intrinsics, ii, jj, False)

    return coords1[..., :2] - coords0, valid


def my_iproj(disps, intrinsics, jacobian=False):
    """pinhole camera inverse projection"""
    ht, wd = disps.shape[-2:]
    fx, fy, cx, cy = extract_intrinsics(intrinsics)

    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float(),
    )

    i = torch.ones_like(disps)
    X = (x - cx) / fx
    Y = (y - cy) / fy
    X = X.expand_as(i)
    Y = Y.expand_as(i)
    pts = torch.stack([X, Y, i, disps], dim=-1)

    if jacobian:
        J = torch.zeros_like(pts)
        J[..., -1] = 1.0
        return pts, J

    return pts, None


def my_proj(Xs, intrinsics, jacobian=False, return_depth=False):
    """pinhole camera projection"""
    fx, fy, cx, cy = extract_intrinsics(intrinsics)
    X, Y, Z, D = Xs.unbind(dim=-1)

    Z = torch.where(Z < 0.5 * MIN_DEPTH, torch.ones_like(Z), Z)
    d = 1.0 / Z

    x = fx * (X * d) + cx
    y = fy * (Y * d) + cy
    if return_depth:
        coords = torch.stack([x, y, D * d], dim=-1)
    else:
        coords = torch.stack([x, y], dim=-1)

    if jacobian:
        B, H, W = d.shape
        o = torch.zeros_like(d)
        proj_jac = torch.stack(
            [
                fx * d,
                o,
                -fx * X * d * d,
                o,
                o,
                fy * d,
                -fy * Y * d * d,
                o,
                # o,     o,    -D*d*d,  d,
            ],
            dim=-1,
        ).view(B, H, W, 2, 4)

        return coords, proj_jac

    return coords, None


def my_actp(Gij, X0, jacobian=False):
    """action on point cloud"""
    X1 = Gij[:, None, None] * X0

    if jacobian:
        X, Y, Z, d = X1.unbind(dim=-1)
        o = torch.zeros_like(d)
        B, H, W = d.shape

        if isinstance(Gij, SE3):
            Ja = torch.stack(
                [
                    d,
                    o,
                    o,
                    o,
                    Z,
                    -Y,
                    o,
                    d,
                    o,
                    -Z,
                    o,
                    X,
                    o,
                    o,
                    d,
                    Y,
                    -X,
                    o,
                    o,
                    o,
                    o,
                    o,
                    o,
                    o,
                ],
                dim=-1,
            ).view(B, H, W, 4, 6)

        elif isinstance(Gij, Sim3):
            Ja = torch.stack(
                [
                    d,
                    o,
                    o,
                    o,
                    Z,
                    -Y,
                    X,
                    o,
                    d,
                    o,
                    -Z,
                    o,
                    X,
                    Y,
                    o,
                    o,
                    d,
                    Y,
                    -X,
                    o,
                    Z,
                    o,
                    o,
                    o,
                    o,
                    o,
                    o,
                    o,
                ],
                dim=-1,
            ).view(B, H, W, 4, 7)

        return X1, Ja

    return X1, None


def ba_pose_jacobian(poses, disps, intrinsics, ii, jj, target, weight):
    """pose jacobian for bundle adjustment"""
    """map points from ii->jj"""
    # inverse project (pinhole)
    X0, _ = my_iproj(disps[ii], intrinsics, jacobian=False)
    # transform
    Gs = SE3(poses)
    Gij = Gs[jj] * Gs[ii].inv()

    X1, Ja = my_actp(Gij, X0, jacobian=True)

    # project (pinhole)
    x1, Jp = my_proj(X1, intrinsics, jacobian=True, return_depth=False)

    # exclude points too close to camera
    valid = ((X1[..., 2] > MIN_DEPTH) & (X0[..., 2] > MIN_DEPTH)).float()
    valid = valid.unsqueeze(-1)

    # Ji transforms according to dual adjoint

    Jj = torch.matmul(Jp, Ja)
    Ji = -Gij[:, None, None, None].adjT(Jj)

    target = target.permute(0, 2, 3, 1)
    weight = weight.permute(0, 2, 3, 1)
    weight = weight * valid

    Jr = (x1 - target) / (x1 - target).norm(dim=-1, keepdim=True)
    Jr = (Jr * weight).unsqueeze(-2)

    Ji = torch.matmul(Jr, Ji)
    Jj = torch.matmul(Jr, Jj)

    Ji = Ji.sum(dim=(1, 2))
    Jj = Jj.sum(dim=(1, 2))

    return Ji, Jj


# def ba_pose_jacobian(poses, disps, intrinsics, ii, jj, target, weight):
#     """pose jacobian for bundle adjustment"""
#     # prepare constants
#     fx, fy, cx, cy = extract_intrinsics(intrinsics)
#     ht, wd = disps.shape[-2:]

#     y, x = torch.meshgrid(
#         torch.arange(ht).to(disps.device).float(),
#         torch.arange(wd).to(disps.device).float(),
#     )

#     # remap frame id
#     ii_unique = torch.unique(ii)

#     ii_to_new = {
#         ii.item(): new_ii.item()
#         for ii, new_ii in zip(ii_unique, torch.arange(ii_unique.shape[0]).to(ii.device))
#     }
#     new_to_ii = {
#         new_ii.item(): ii.item()
#         for ii, new_ii in zip(ii_unique, torch.arange(ii_unique.shape[0]).to(ii.device))
#     }  # in fact it is ii_unique itself

#     new_ii = torch.zeros_like(ii)
#     new_jj = torch.zeros_like(jj)

#     for i in range(new_ii.shape[0]):
#         new_ii[i] = ii_to_new[ii[i].item()]
#         new_jj[i] = ii_to_new[jj[i].item()]

#     # new_ii = new_ii[]
#     # new_jj = new_jj[:2]
#     # target = target[:2]
#     # weight = weight[:2]

#     new_poses = poses[ii_unique]
#     new_disps = disps[ii_unique]

#     # re-parameterate new_poses

#     reparameterized_poses = torch.zeros(new_poses.shape[0], 6).to(new_poses.device)
#     reparameterized_poses[:, :3] = lietorch.SE3(new_poses).translation()[:, :3]
#     reparameterized_poses[:, 3:] = lietorch.SE3(new_poses).log()[:, 3:]

#     # jacobian = torch.zeros(E, 2, H, W, new_poses.shape[0], 6).to(new_poses.device)

#     # TODO: edge wise jacobian computation
#     i = torch.ones_like(new_disps)
#     X = (x - cx) / fx
#     Y = (y - cy) / fy
#     X = X.expand_as(i)
#     Y = Y.expand_as(i)
#     pts = torch.stack([X / new_disps, Y / new_disps, i / new_disps, i], dim=-1)
#     pts_i = pts[new_ii]

#     # pose_jacobians = []

#     def projective_functor(reparameterized_poses):
#         # used outer parameters and constants
#         Rs_ii = lietorch.SO3.exp(reparameterized_poses[:, 3:]).matrix()
#         ts_ii = reparameterized_poses[:, :3]

#         Rs_jj = lietorch.SO3.exp(reparameterized_poses[:, 3:]).matrix()
#         ts_jj = reparameterized_poses[:, :3]

#         Rs_ii = Rs_ii[new_ii]
#         ts_ii = ts_ii[new_ii]
#         Rs_jj = Rs_jj[new_jj]
#         ts_jj = ts_jj[new_jj]

#         Rs_ij = torch.bmm(Rs_jj, Rs_ii.transpose(-1, -2))
#         T_ij = torch.zeros_like(Rs_ij)
#         T_ij[:, :3, :3] = Rs_ij[:, :3, :3]

#         T_ij[:, :3, 3:4] = (
#             -torch.bmm(Rs_ij[:, :3, :3], ts_ii[..., None]) + ts_jj[..., None]
#         )

#         pts_j = torch.bmm(pts_i.reshape(pts_i.shape[0], -1, 4), T_ij.transpose(-1, -2))
#         pts_j = pts_j.reshape(*pts_i.shape)

#         xp = fx * pts_j[..., 0] + cx
#         yp = fy * pts_j[..., 1] + cy
#         zp = pts_j[..., 2]

#         xp = xp / zp
#         yp = yp / zp

#         valid = torch.logical_and(zp > 0.2, zp < 4.0).float().unsqueeze(1)

#         coords1 = torch.stack([xp, yp], dim=1)

#         loss = valid * (target - coords1) * weight
#         return loss

#     jacobian = torch.autograd.functional.jacobian(
#         projective_functor,
#         (reparameterized_poses),
#     )

#     return jacobian
