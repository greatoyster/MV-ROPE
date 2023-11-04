import numpy as np
import torch
import lietorch
import droid_backends

from torch.multiprocessing import Process, Queue, Lock, Value
from collections import OrderedDict

from droid_net import cvx_upsample
import geom.projective_ops as pops
import geom.utils as gutils
import geom.pose_averaging as pose_avg

try:
    import ipdb as pdb
except:
    import pdb

from rich import print


class DepthVideo:
    def __init__(self, image_size=[480, 640], buffer=1024, stereo=False, device="cuda"):
        # object level setup
        self.max_obj_num = 10

        # current keyframe count
        self.counter = Value("i", 0)
        self.ready = Value("i", 0)
        self.ht = ht = image_size[0]
        self.wd = wd = image_size[1]

        ### state attributes ###
        self.tstamp = torch.zeros(
            buffer, device="cuda", dtype=torch.float
        ).share_memory_()
        self.images = torch.zeros(buffer, 3, ht, wd, device="cuda", dtype=torch.uint8)
        self.dirty = torch.zeros(
            buffer, device="cuda", dtype=torch.bool
        ).share_memory_()
        self.red = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.poses = torch.zeros(
            buffer, 7, device="cuda", dtype=torch.float
        ).share_memory_()
        self.disps = torch.ones(
            buffer, ht // 8, wd // 8, device="cuda", dtype=torch.float
        ).share_memory_()  # optimized depth map
        self.disps_sens = torch.zeros(
            buffer, ht // 8, wd // 8, device="cuda", dtype=torch.float
        ).share_memory_()  # depth camera input
        self.disps_up = torch.zeros(
            buffer, ht, wd, device="cuda", dtype=torch.float
        ).share_memory_()  # upsampled depth map
        self.intrinsics = torch.zeros(
            buffer, 4, device="cuda", dtype=torch.float
        ).share_memory_()

        self.stereo = stereo
        c = 1 if not self.stereo else 2

        ### feature attributes ###
        self.fmaps = torch.zeros(
            buffer, c, 128, ht // 8, wd // 8, dtype=torch.half, device="cuda"
        ).share_memory_()
        self.nets = torch.zeros(
            buffer, 128, ht // 8, wd // 8, dtype=torch.half, device="cuda"
        ).share_memory_()
        self.inps = torch.zeros(
            buffer, 128, ht // 8, wd // 8, dtype=torch.half, device="cuda"
        ).share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor(
            [0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cuda"
        )
        self.pose_covariances = torch.zeros(
            buffer, 6, 6, dtype=torch.float, device="cuda"
        ).share_memory_()
        self.pose_jacobians = torch.zeros(
            buffer, 2, 6, dtype=torch.float, device="cuda"
        ).share_memory_()

        # TODO: add new attributes for object pose estimation
        self.obj_poses = torch.zeros(
            buffer, self.max_obj_num, 8, device="cuda", dtype=torch.float
        ).share_memory_()
        self.active_objs = torch.zeros(
            buffer, self.max_obj_num, device="cuda", dtype=torch.uint8
        ).share_memory_()
        self.obj_pose_scores = torch.zeros(
            buffer, self.max_obj_num, 1, device="cuda", dtype=torch.float
        )

        self.obj_masks = torch.zeros(
            buffer, ht, wd, device="cuda", dtype=torch.uint8
        ).share_memory_()
        self.nocs = torch.zeros(
            buffer, 3, ht, wd, device="cuda", dtype=torch.float
        ).share_memory_()
        self.metas = [None] * buffer

        self.gt_obj_poses = torch.zeros(
            buffer, self.max_obj_num, 8, device="cuda", dtype=torch.float
        ).share_memory_()

        self.evalation_statistics = dict(
            pose_count=0,
            deg10_trans10=0,
            deg10_trans5=0,
            deg5_trans10=0,
            deg5_trans5=0,
            iou25=0,
            iou50=0,
            iou75=0,
        )

        self.symmetric_categories = [1, 2, 4]

        self.shared_grid = torch.cartesian_prod(torch.arange(ht), torch.arange(wd)).to(
            device="cuda"
        )
        self.shared_grid = self.shared_grid.view(ht, wd, 2).permute(2, 0, 1)

        self.boundary_threshold = 10
        self.topk = 10

        # Here we choose top 10 ransac score to perform pose averaging
        self.global_active_objs = torch.zeros(
            self.max_obj_num, self.topk, device="cuda", dtype=torch.uint8
        ).share_memory_()
        self.observations = torch.zeros(
            self.max_obj_num, self.topk, 7 + 8 + 1, device="cuda", dtype=torch.float
        ).share_memory_()
        self.global_obj_poses = torch.zeros(
            self.max_obj_num, 8, device="cuda", dtype=torch.float
        ).share_memory_()

        self.box_scales = torch.zeros(
            self.max_obj_num, 3, device="cuda", dtype=torch.float
        ).share_memory_()

    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1

        elif (
            isinstance(index, torch.Tensor) and index.max().item() > self.counter.value
        ):
            self.counter.value = index.max().item() + 1

        # self.dirty[index] = True
        self.tstamp[index] = item[0]
        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]

        if item[4] is not None:
            depth = item[4][3::8, 3::8]
            self.disps_sens[index] = torch.where(depth > 0, 1.0 / depth, depth)

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6:
            self.fmaps[index] = item[6]

        if len(item) > 7:
            self.nets[index] = item[7]

        if len(item) > 8:
            self.inps[index] = item[8]

        if len(item) <= 9:
            print("strange item length:", len(item))

        if len(item) > 9:
            self.obj_masks[index] = item[9]

        if len(item) > 10:
            self.nocs[index] = item[10]

        if len(item) > 11:
            self.metas[index] = item[11]

        if len(item) > 12:
            for k, v in item[12].items():
                self.gt_obj_poses[index][k] = gutils.mat_to_sim3_tensor(v)

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """index the depth video"""

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index],
            )

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)

    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj):
        """to device, long, {-1}"""

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """upsample disparity"""

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()

    def normalize(self):
        """normalize depth and poses"""

        with self.get_lock():
            s = self.disps[: self.counter.value].mean()
            self.disps[: self.counter.value] /= s
            self.poses[: self.counter.value, :3] *= s
            self.dirty[: self.counter.value] = True

    def reproject(self, ii, jj):
        """project points from ii -> jj"""
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = lietorch.SE3(self.poses[None])

        coords, valid_mask = pops.projective_transform(
            Gs, self.disps[None], self.intrinsics[None], ii, jj
        )

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """frame distance metric"""

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))

        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:
            poses = self.poses[: self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta
            )

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta
            )

            d = 0.5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta
            )

        if return_matrix:
            return d.reshape(N, N)

        return d

    # TODO: change this ba interface for object pose estimation
    def ba(
        self,
        target,
        weight,
        eta,
        ii,
        jj,
        t0=1,
        t1=None,
        itrs=2,
        lm=1e-4,
        ep=0.1,
        motion_only=False,
        obj_enabled=False,
    ):
        """dense bundle adjustment (DBA)"""
        # target: optical flow between ii and jj
        # weight: weight of optical flow

        with self.get_lock():
            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1
            droid_backends.ba(
                self.poses,
                self.disps,
                self.intrinsics[0],
                self.disps_sens,
                target,  # Ex2xHxW
                weight,  # Ex2xHxW
                eta,
                ii,
                jj,
                t0,
                t1,
                itrs,
                lm,
                ep,
                motion_only,
            )

            self.disps.clamp_(min=0.001)

            # TODO: add multi frame nocs fusion && dump optical flow here

            is_keyframe_refined = obj_enabled

            if is_keyframe_refined:
                # TODO: calculate pose jacobian here:
                Ji, Jj = pops.ba_pose_jacobian(
                    self.poses, self.disps, self.intrinsics[0], ii, jj, target, weight
                )
                uii = torch.unique(ii)
                self.pose_jacobians[uii] = 0
                for i, j in zip(ii, jj):
                    self.pose_jacobians[i] += Ji[i]
                    self.pose_jacobians[j] += Jj[j]
                for i in uii:
                    self.pose_covariances[i] = (
                        self.pose_jacobians[i].t() @ self.pose_jacobians[i]
                    )
                    self.pose_covariances[i] = torch.linalg.inv(
                        self.pose_covariances[i]
                    )

    def object_pose_averaging(self):
        with self.get_lock():
            for obj_id in range(self.max_obj_num):
                if self.global_active_objs[obj_id].sum() == 0:
                    continue

                rotations = []
                scales = []
                transes = []
                for k in range(self.topk):
                    if self.observations[obj_id, k, -1] < 1e-5:
                        continue
                    # 6d only
                    camera_to_world = (
                        lietorch.SE3(self.observations[obj_id, k, :7]).inv().matrix()
                    )
                    nocs_to_camera = lietorch.SE3(
                        self.observations[obj_id, k, 7 : 7 + 7]
                    ).matrix()
                    nocs_to_world = camera_to_world @ nocs_to_camera

                    rotations.append(nocs_to_world[:3, :3])
                    scales.append(self.observations[obj_id, k, 7 + 7].view(-1))
                    transes.append(nocs_to_world[:3, 3].view(-1))

                rotations = torch.stack(rotations)  # nocs to world
                scales = torch.stack(scales)
                transes = torch.stack(transes)

                robust_rotation = pose_avg.so3_chordal_l1_mean(
                    rotations, False, 10, 0.001
                ).cuda()
                robust_scale = torch.median(scales, dim=0).values
                robust_trans = torch.median(transes, dim=0).values
                self.global_obj_poses[obj_id] = gutils.Rst_to_sim3_tensor(
                    robust_rotation, robust_scale, robust_trans
                )

    def update_topk_object_poses(self, fid, obj_id, score):
        update = False
        with self.get_lock():
            # check activate of that object
            if self.global_active_objs[obj_id].sum() < self.topk:
                ix = torch.nonzero(self.global_active_objs[obj_id] == 0)[0].item()
                self.global_active_objs[obj_id, ix] = 1
                self.observations[obj_id, ix, :7] = self.poses[fid]
                self.observations[obj_id, ix, 7 : 7 + 8] = self.obj_poses[fid, obj_id]
                self.observations[obj_id, ix, -1] = score
                update = True
            else:
                ix = torch.argmin(self.observations[obj_id, :, -1]).item()
                min_score = self.observations[obj_id, ix, -1]
                if score > min_score:
                    self.observations[obj_id, ix, :7] = self.poses[fid]
                    self.observations[obj_id, ix, 7 : 7 + 8] = self.obj_poses[
                        fid, obj_id
                    ]
                    self.observations[obj_id, ix, -1] = score
                    update = True
        return update

    def object_centric_ba(self, ii, evaluation=False):
        with self.get_lock():
            fids = torch.unique(ii)
            tstamps = torch.index_select(self.tstamp, 0, fids)
            # print(tstamps)
            obj_centric_poses = {}
            frame_centric_poses = {}
            fx, fy, cx, cy = (
                self.intrinsics[0][0],
                self.intrinsics[0][1],
                self.intrinsics[0][2],
                self.intrinsics[0][3],
            )

            #  remove object in edge area
            ht, wd = self.obj_masks.shape[:2]
            # compute each object pose in each frame
            for fid in fids:
                objs = self.obj_masks[fid].unique()[:-1].tolist()
                for obj_id in objs:
                    # skip estimated keyframes
                    if self.active_objs[fid, obj_id] == 1:
                        continue
                    # estimate object pose in downsampled space
                    # if obj_id != 3:
                    #     continue
                    obj_mask = self.obj_masks[fid]
                    obj_mask = obj_mask == obj_id
                    nonzero_indices = torch.nonzero(obj_mask)
                    nonzero_indices_y, nonzero_indices_x = (
                        nonzero_indices[:, 0],
                        nonzero_indices[:, 1],
                    )

                    if nonzero_indices.numel() < 200:
                        print(
                            f"warning: we can not detect object with id {obj_id}, too few pixels"
                        )
                        continue
                    obj_nocs = (
                        self.nocs[fid][:, nonzero_indices_y, nonzero_indices_x]
                    ).view([3, -1])
                    obj_depth = (
                        self.disps_up[fid][nonzero_indices_y, nonzero_indices_x]
                    ).view([1, -1])
                    obj_grid = (
                        self.shared_grid[:, nonzero_indices_y, nonzero_indices_x]
                    ).view(2, 1, -1)

                    obj_grid_y, obj_grid_x = obj_grid.unbind(0)
                    obj_X = (obj_grid_x - cx * 8) / (fx * 8)
                    obj_Y = (obj_grid_y - cy * 8) / (fy * 8)
                    obj_I = torch.ones_like(obj_depth)
                    obj_xyz = torch.cat([obj_X, obj_Y, obj_I], dim=0)
                    obj_xyz /= obj_depth
                    obj_nocs = obj_nocs.permute(1, 0).unsqueeze(0)
                    obj_xyz = obj_xyz.permute(1, 0).unsqueeze(0)
                    # add ransac for robust estimation
                    transf, score = gutils.fast_umeyama_ransac(
                        obj_nocs, obj_xyz, 600, 0.015
                    )

                    inlier_ratio = score / obj_nocs.size(1)
                    self.obj_pose_scores[fid, obj_id] = inlier_ratio

                    # gather the estimated pose of each object among all frames, and average its
                    if obj_centric_poses.get(obj_id, None) != None:
                        obj_centric_poses[obj_id].append((fid, transf))
                    else:
                        obj_centric_poses[obj_id] = [(fid, transf)]

                    if frame_centric_poses.get(fid) != None:
                        frame_centric_poses[fid].append((obj_id, transf))
                    else:
                        frame_centric_poses[fid] = [(obj_id, transf)]

                    rot, t, s = (
                        transf.R.transpose(-1, -2),
                        transf.T,
                        transf.s.unsqueeze(0),
                    )  # nocs to camera
                    sim3_tensor = gutils.Rst_to_sim3_tensor(rot, s, t)
                    self.obj_poses[fid, obj_id] = sim3_tensor[:]
                    self.active_objs[fid, obj_id] = 1

                    update = self.update_topk_object_poses(fid, obj_id, inlier_ratio)
                    if update:
                        self.box_scales[obj_id, 0] = max(
                            self.box_scales[obj_id, 0],
                            torch.max(obj_nocs[0, :, 0].abs()),
                        )
                        self.box_scales[obj_id, 1] = max(
                            self.box_scales[obj_id, 1],
                            torch.max(obj_nocs[0, :, 1].abs()),
                        )
                        self.box_scales[obj_id, 2] = max(
                            self.box_scales[obj_id, 2],
                            torch.max(obj_nocs[0, :, 2].abs()),
                        )

            self.object_pose_averaging()

            for obj_id, stamped_poses in obj_centric_poses.items():
                for sp in stamped_poses:
                    fid, transf = sp
                    if evaluation:
                        if self.gt_obj_poses[fid, obj_id][-1] < 1e-3:
                            print(
                                f"warning skip evaluate instance {obj_id} of frame {self.tstamp[fid]}"
                            )
                            continue
                        est = lietorch.Sim3(self.obj_poses[fid, obj_id])
                        gt = lietorch.Sim3(self.gt_obj_poses[fid, obj_id])
                        t_diff = 0
                        rot_diff = 0
                        iou = 0
                        cat_id = self.metas[fid][obj_id]
                        if cat_id in self.symmetric_categories:
                            iou = gutils.compute_3d_IoU(est.matrix(), gt.matrix(), True)
                            rot_diff, t_diff = gutils.compute_RT_errors(
                                est.matrix(), gt.matrix(), True
                            )
                            # print(f"iou of sym object {obj_id} = {iou}")
                            # print(f"rotation error of sym object {obj_id} = {rot_diff}")
                            # print(f"translation error of sym object {obj_id} = {t_diff}")
                        else:
                            iou = gutils.compute_3d_IoU(est.matrix(), gt.matrix())
                            rot_diff, t_diff = gutils.compute_RT_errors(
                                est.matrix(), gt.matrix()
                            )
                            # print(f"iou of object {obj_id} = {iou}")
                            # print(f"rotation error of object {obj_id} = {rot_diff}")
                            # print(f"translation error of object {obj_id} = {t_diff}")
