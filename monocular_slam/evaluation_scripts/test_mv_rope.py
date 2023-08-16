import sys
import os

sys.path.append("droid_slam")
import glob
import seaborn as sns
import torch.nn.functional as F
import pickle
import argparse
import cv2
import torch
import numpy as np
from tqdm import tqdm
from droid import Droid


try:
    import ipdb as pdb
except:
    import pdb


nocs_color_map = {k + 1: v for k, v in enumerate(sns.color_palette())}


def save_reconstruction(droid, reconstruction_path):
    from pathlib import Path
    import random
    import string

    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    images = droid.video.images[:t].cpu().numpy()
    disps = droid.video.disps_up[:t].cpu().numpy()
    poses = droid.video.poses[:t].cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()
    nocs = droid.video.nocs[:t].cpu().numpy()
    masks = droid.video.obj_masks[:t].cpu().numpy()

    gt_obj_poses = droid.video.gt_obj_poses[:t].cpu().numpy()
    obj_poses = droid.video.obj_poses.cpu().numpy()
    active_objs = droid.video.active_objs.cpu().numpy()
    obj_pose_scores = droid.video.obj_pose_scores.cpu().numpy()

    global_obj_poses = droid.video.global_obj_poses.cpu().numpy()
    box_scales = droid.video.box_scales.cpu().numpy()

    Path("reconstructions/{}".format(reconstruction_path)).mkdir(
        parents=True, exist_ok=True
    )
    np.save("reconstructions/{}/tstamps.npy".format(reconstruction_path), tstamps)
    np.save("reconstructions/{}/images.npy".format(reconstruction_path), images)
    np.save("reconstructions/{}/disps.npy".format(reconstruction_path), disps)
    np.save("reconstructions/{}/poses.npy".format(reconstruction_path), poses)
    np.save("reconstructions/{}/intrinsics.npy".format(reconstruction_path), intrinsics)
    np.save("reconstructions/{}/nocs.npy".format(reconstruction_path), nocs)
    np.save("reconstructions/{}/masks.npy".format(reconstruction_path), masks)
    np.save(
        "reconstructions/{}/gt_obj_poses.npy".format(reconstruction_path), gt_obj_poses
    )
    np.save("reconstructions/{}/obj_poses.npy".format(reconstruction_path), obj_poses)
    np.save(
        "reconstructions/{}/active_objs.npy".format(reconstruction_path), active_objs
    )
    np.save(
        "reconstructions/{}/obj_pose_scores.npy".format(reconstruction_path),
        obj_pose_scores,
    )

    np.save("reconstructions/{}/global_obj_poses.npy".format(reconstruction_path), global_obj_poses)
    np.save("reconstructions/{}/box_scales.npy".format(reconstruction_path), box_scales)

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow("image", image / 255.0)
    cv2.waitKey(1)


def show_mask(mask):
    mask = mask.cpu().numpy()
    mask_color = np.zeros([mask.shape[0], mask.shape[1], 3])
    inst_ids = np.unique(mask)
    inst_ids = inst_ids[1:] if inst_ids[0] == 0 else inst_ids
    inst_ids = inst_ids[:-1] if inst_ids[-1] == 255 else inst_ids

    for inst_id in inst_ids:
        mask_color[mask == inst_id] = nocs_color_map[inst_id]
        mask_color[mask == inst_id] *= 255
    # TODO: add coloring code to bit mask, in our case, each instance should have one unique color
    cv2.imshow("mask", mask_color.astype(np.uint8))
    cv2.waitKey(1)


def show_nocs(nocs):
    nocs = nocs.cpu().permute(1, 2, 0).numpy() + 0.5
    nocs[:, :, 2] = 1 - nocs[:, :, 2]
    cv2.imshow("nocs", nocs)
    cv2.waitKey(1)


def image_stream(datapath, use_depth=False, use_aux=True, stride=1):
    """image generator"""

    fx, fy, cx, cy = np.loadtxt(os.path.join(datapath, "calibration.txt")).tolist()
    image_list = sorted(glob.glob(os.path.join(datapath, "*_color.png")))[::stride]
    depth_list = sorted(glob.glob(os.path.join(datapath, "*_depth.png")))[::stride]
    mask_list = sorted(glob.glob(os.path.join(datapath, "*_mask.pred.png")))[::stride]
    nocs_list = sorted(glob.glob(os.path.join(datapath, "*_coord.pred.png")))[::stride]
    meta_list = sorted(glob.glob(os.path.join(datapath, "*_meta.pred.txt")))[::stride]
    gt_obj_poses_list = sorted(glob.glob(os.path.join(datapath, "*.remake.pkl")))[
        ::stride
    ]  # [results**, *.remake.pkl]

    print(len(image_list))
    print(len(depth_list))
    print(len(nocs_list))
    print(len(mask_list))
    print(len(meta_list))
    print(len(gt_obj_poses_list))

    assert len(image_list) == len(depth_list)
    assert len(image_list) == len(mask_list)
    assert len(image_list) == len(nocs_list)
    assert len(image_list) == len(meta_list)
    assert len(image_list) == len(gt_obj_poses_list)

    for t, (image_file, depth_file) in enumerate(zip(image_list, depth_list)):
        # print(image_file)
        image = cv2.imread(image_file)
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH) / 1000.0

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((192 * 256 * 4) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((192 * 256 * 4) / (h0 * w0)))

        # h1 = int(h0 * np.sqrt((192 * 256) / (h0 * w0)))
        # w1 = int(w0 * np.sqrt((192 * 256) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[: h1 - h1 % 8, : w1 - w1 % 8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        depth = torch.as_tensor(depth)
        depth = F.interpolate(depth[None, None], (h1, w1)).squeeze()
        depth = depth[: h1 - h1 % 8, : w1 - w1 % 8]

        aux_data = None

        if use_aux:
            with open(meta_list[t], "r") as f:
                # [inst_id(1-indexed), cat_id(0-indexed), sym_name)
                meta = [line.strip().split(" ") for line in f.readlines()]
            meta = {int(m[0]): int(m[1]) for m in meta if len(m) == 3}
            mask = cv2.imread(mask_list[t])[:, :, 2]

            mask = cv2.resize(mask, (w1, h1), interpolation=cv2.INTER_NEAREST)
            mask = torch.as_tensor(mask)

            nocs = cv2.imread(nocs_list[t], -1)
            nocs = nocs[:, :, [2, 1, 0]]
            nocs = cv2.resize(nocs, (w1, h1), interpolation=cv2.INTER_NEAREST)
            nocs = nocs.astype(np.float32) / 255.0
            nocs[:, :, 2] = 1 - nocs[:, :, 2]
            nocs -= 0.5
            nocs = torch.as_tensor(nocs).permute(2, 0, 1)
            gt_obj_poses = dict()

            with open(gt_obj_poses_list[t], "rb") as f:
                data = pickle.load(f)
                for k, v in data.items():
                    gt_obj_poses[k] = torch.as_tensor(v).float().cuda()
            # print(gt_obj_poses)
            aux_data = (mask, nocs, meta, gt_obj_poses)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= w1 / w0
        intrinsics[1::2] *= h1 / h0


        if use_depth:
            yield t, image[None], depth, intrinsics, aux_data

        else:
            yield t, image[None], intrinsics, aux_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath")
    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--buffer", type=int, default=1024)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--filter_thresh", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--keyframe_thresh", type=float, default=3.5)
    parser.add_argument("--frontend_thresh", type=float, default=16.0)
    parser.add_argument("--frontend_window", type=int, default=16)
    parser.add_argument("--frontend_radius", type=int, default=1)
    parser.add_argument("--frontend_nms", type=int, default=0)

    parser.add_argument("--stereo", action="store_true")
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--nocs", action="store_true")
    parser.add_argument("--mask", action="store_true")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", type=bool, default=True)
    parser.add_argument("--zero_depth", action="store_true")
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    parser.add_argument("--statistics_path", help="path to object pose statistics")
    parser.add_argument("--trajectory_path", help="path to object pose trajectory")

    args = parser.parse_args()

    torch.multiprocessing.set_start_method("spawn")

    print("Running evaluation on {}".format(args.datapath))
    print(args)

    droid = None

    # this can usually be set to 2-3 except for "camera_shake" scenes
    # set to 2 for test scenes
    stride = 1
    use_aux = args.mask and args.nocs
    tstamps = []
    if args.depth:
        for t, image, depth, intrinsics, aux_data in tqdm(
            image_stream(args.datapath, use_aux=use_aux, use_depth=True, stride=stride)
        ):
            mask, nocs, meta, gt_obj_poses = aux_data
            if not args.disable_vis:
                show_image(image[0])
                show_mask(mask)
                show_nocs(nocs)

            if t == 0:
                args.image_size = [image.shape[2], image.shape[3]]
                droid = Droid(args)
            
            if args.zero_depth:
                depth.zero_()
            
            droid.track(
                t,
                image,
                depth,
                intrinsics=intrinsics,
                mask=mask,
                coord=nocs,
                meta=meta,
                gt_obj_poses=gt_obj_poses,
            )
        cv2.destroyAllWindows()
        if args.reconstruction_path is not None:
            save_reconstruction(droid, args.reconstruction_path)
        if args.statistics_path != None:
            with open(args.statistics_path, "wb") as f:
                pickle.dump(droid.video.evalation_statistics, f)

        # traj_est = droid.terminate(image_stream(
        #     args.datapath, use_depth=False, stride=stride))

        # if args.trajectory_path != None:
        #     with open(args.trajectory_path, "wb") as f:
        #         pickle.dump(traj_est, f)
    else:
        raise NotImplementedError
        for t, image, intrinsics, _ in tqdm(
            image_stream(args.datapath, use_depth=False, stride=stride)
        ):
            if not args.disable_vis:
                show_image(image[0])

            if t == 0:
                args.image_size = [image.shape[2], image.shape[3]]
                droid = Droid(args)

            droid.track(t, image, intrinsics=intrinsics)

        traj_est = droid.terminate(
            image_stream(args.datapath, use_depth=False, stride=stride)
        )

    ### run evaluation ###

    # print("#"*20 + " Results...")

    # import evo
    # from evo.core.trajectory import PoseTrajectory3D
    # from evo.tools import file_interface
    # from evo.core import sync
    # import evo.main_ape as main_ape
    # from evo.core.metrics import PoseRelation

    # image_path = os.path.join(args.datapath, 'rgb')
    # images_list = sorted(glob.glob(os.path.join(image_path, '*.png')))[::stride]
    # tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]

    # traj_est = PoseTrajectory3D(
    #     positions_xyz=traj_est[:,:3],
    #     orientations_quat_wxyz=traj_est[:,3:],
    #     timestamps=np.array(tstamps))

    # gt_file = os.path.join(args.datapath, 'groundtruth.txt')
    # traj_ref = file_interface.read_tum_trajectory_file(gt_file)

    # traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    # result = main_ape.ape(traj_ref, traj_est, est_name='traj',
    #     pose_relation=PoseRelation.translation_part, align=True, correct_scale=False)

    # print(result.stats)
