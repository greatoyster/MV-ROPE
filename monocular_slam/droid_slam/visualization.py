import torch
import cv2
import lietorch
import droid_backends
import time
import argparse
import numpy as np
import open3d as o3d
from rich import print
from lietorch import SE3, Sim3
import geom.projective_ops as pops

CAM_POINTS = np.array(
    [
        [0, 0, 0],
        [-1, -1, 1.5],
        [1, -1, 1.5],
        [1, 1, 1.5],
        [-1, 1, 1.5],
        [-0.5, 1, 1.5],
        [0.5, 1, 1.5],
        [0, 1.2, 1.5],
    ]
)

CAM_LINES = np.array(
    [[1, 2], [2, 3], [3, 4], [4, 1], [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]]
)


def generate_color(time):
    # Adjust the color properties based on time
    frequency = 0.005  # Controls the speed of color change
    phase_shift = 0  # Controls the starting point of the color change
    amplitude = 127  # Controls the range of color variation

    # Generate the RGB values using sine waves
    red = (
        int(
            amplitude * np.sin(frequency * time + 2 * np.pi * 0 / 3 + phase_shift) + 128
        )
        / 255
    )
    green = (
        int(
            amplitude * np.sin(frequency * time + 2 * np.pi * 1 / 3 + phase_shift) + 128
        )
        / 255
    )
    blue = (
        int(
            amplitude * np.sin(frequency * time + 2 * np.pi * 2 / 3 + phase_shift) + 128
        )
        / 255
    )

    return (red, green, blue)


def white_balance(img):
    # from https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - (
        (avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1
    )
    result[:, :, 2] = result[:, :, 2] - (
        (avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1
    )
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def create_camera_actor(g, scale=0.05):
    """build open3d camera polydata"""
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES),
    )

    color = (g * 1.0, 0.5 * (1 - g), 0.9 * (1 - g))
    camera_actor.paint_uniform_color(color)
    return camera_actor


def create_point_actor(points, colors):
    """open3d point cloud from numpy array"""
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def create_cube_actor():
    lineset = o3d.geometry.LineSet()
    vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    lineset.points = o3d.utility.Vector3dVector(vertices)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.translate(np.array([-0.5, -0.5, -0.5]))
    lineset.paint_uniform_color((0.0, 1.0, 1.0))
    return lineset


def create_coord_frame_actor():
    o3d.geometry.TriangleMesh()
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3, origin=np.array([0.0, 0, 0.0])
    )
    return coord


def droid_visualization(video, device="cuda:0"):
    """DROID visualization frontend"""

    torch.cuda.set_device(device)
    droid_visualization.video = video
    droid_visualization.cameras = {}
    droid_visualization.points = {}
    droid_visualization.coord_frames = {}
    droid_visualization.bbox3ds = {}

    droid_visualization.warmup = 8
    droid_visualization.scale = 1.0
    droid_visualization.ix = 0
    droid_visualization.obj_pose_mode = True
    droid_visualization.world_axis = None

    droid_visualization.filter_thresh = 0.005
    droid_visualization.color_counter = 0

    def set_obj_pose_mode(vis):
        print("object pose mode triggered")
        droid_visualization.obj_pose_mode = not droid_visualization.obj_pose_mode
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[
                : droid_visualization.video.counter.value
            ] = True

    def increase_filter(vis):
        droid_visualization.filter_thresh *= 2
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[
                : droid_visualization.video.counter.value
            ] = True

    def decrease_filter(vis):
        droid_visualization.filter_thresh *= 0.5
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[
                : droid_visualization.video.counter.value
            ] = True

    def export_scene_points(vis):
        print("Saving scene point cloud and bounding box")
        scene_points = o3d.geometry.PointCloud()
        bboxes = o3d.geometry.LineSet()
        for box3d in droid_visualization.bbox3ds.values():
            bboxes += box3d 
        o3d.io.write_line_set("bboxes.ply", bboxes)
        for pc in droid_visualization.points.values():
            scene_points += pc
        o3d.io.write_point_cloud("scene_points.ply", scene_points)

    def animation_callback(vis):
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()

        color = generate_color(droid_visualization.color_counter)
        droid_visualization.color_counter += 1

        with torch.no_grad():
            # if droid_visualization.world_axis == None:
            #     droid_visualization.world_axis = create_coord_frame_actor()
            #     vis.add_geometry(droid_visualization.world_axis)

            with video.get_lock():
                t = video.counter.value
                (dirty_index,) = torch.where(video.dirty.clone())
                dirty_index = dirty_index

            if len(dirty_index) == 0:
                return

            video.dirty[dirty_index] = False

            global_obj_poses = video.global_obj_poses  # [max_obj_num, topk, 8]
            # [max_obj_num, topk]
            global_active_objs = video.global_active_objs
            box_scales = video.box_scales

            # convert poses to 4x4 matrix
            poses = torch.index_select(video.poses, 0, dirty_index)
            disps = torch.index_select(video.disps, 0, dirty_index)
            Ps = SE3(poses).inv().matrix().cpu().numpy()

            images = torch.index_select(video.images, 0, dirty_index)
            images = images.cpu()[:, [2, 1, 0], 3::8, 3::8].permute(0, 2, 3, 1) / 255.0
            points = droid_backends.iproj(
                SE3(poses).inv().data, disps, video.intrinsics[0]
            ).cpu()  # poses: world to camera

            thresh = droid_visualization.filter_thresh * torch.ones_like(
                disps.mean(dim=[1, 2])
            )

            count = droid_backends.depth_filter(
                video.poses, video.disps, video.intrinsics[0], dirty_index, thresh
            )

            count = count.cpu()
            disps = disps.cpu()
            masks = (count >= 2) & (disps > 0.5 * disps.mean(dim=[1, 2], keepdim=True))

            for i in range(len(dirty_index)):
                pose = Ps[i]
                ix = dirty_index[i].item()

                if ix in droid_visualization.cameras:
                    vis.remove_geometry(droid_visualization.cameras[ix])
                    del droid_visualization.cameras[ix]

                if ix in droid_visualization.points:
                    vis.remove_geometry(droid_visualization.points[ix])
                    del droid_visualization.points[ix]

                # if droid_visualization.obj_pose_mode:
                #     if ix in droid_visualization.coord_frames:
                #         for geo_obj in droid_visualization.coord_frames[ix]:
                #             vis.remove_geometry(geo_obj)
                #         del droid_visualization.coord_frames[ix]

                #     if ix in droid_visualization.bbox3ds:
                #         for geo_obj in droid_visualization.bbox3ds[ix]:
                #             vis.remove_geometry(geo_obj)
                #         del droid_visualization.bbox3ds[ix]

                ### add camera actor ###
                cam_actor = create_camera_actor(True)
                cam_actor.transform(pose)
                cam_actor.paint_uniform_color(color)

                vis.add_geometry(cam_actor)
                droid_visualization.cameras[ix] = cam_actor

                mask = masks[i].reshape(-1)
                pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
                clr = images[i].reshape(-1, 3)[mask].cpu().numpy()

                ## add point actor ###
                point_actor = create_point_actor(pts, clr)
                vis.add_geometry(point_actor)
                droid_visualization.points[ix] = point_actor

            # visualized global object poses
            if droid_visualization.obj_pose_mode:
                max_obj_num = global_obj_poses.size(0)
                for obj_id in range(max_obj_num):
                    if global_active_objs[obj_id].sum() == 0:
                        continue
                    if obj_id in droid_visualization.coord_frames:
                        vis.remove_geometry(droid_visualization.coord_frames[obj_id])
                        del droid_visualization.coord_frames[obj_id]
                    if obj_id in droid_visualization.bbox3ds:
                        vis.remove_geometry(droid_visualization.bbox3ds[obj_id])
                        del droid_visualization.bbox3ds[obj_id]

                    nocs_to_world = (
                        Sim3(global_obj_poses[obj_id]).matrix().cpu().numpy()
                    )

                    obj_coord_frame = create_coord_frame_actor()
                    obj_coord_frame.transform(nocs_to_world)

                    # add obj_bbox3d scale for visualizaiton
                    scale_transform = np.eye(4)
                    scale_transform[0, 0] = 2 * box_scales[obj_id, 0]
                    scale_transform[1, 1] = 2 * box_scales[obj_id, 1]
                    scale_transform[2, 2] = 2 * box_scales[obj_id, 2]

                    obj_bbox3d = create_cube_actor()

                    obj_bbox3d.transform(scale_transform)
                    obj_bbox3d.transform(nocs_to_world)
                    obj_bbox3d.paint_uniform_color(color)

                    vis.add_geometry(obj_coord_frame)
                    vis.add_geometry(obj_bbox3d)

                    droid_visualization.coord_frames[obj_id] = obj_coord_frame
                    droid_visualization.bbox3ds[obj_id] = obj_bbox3d

            # hack to allow interacting with vizualization during inference
            # print(cam.extrinsic)
            if len(droid_visualization.cameras) >= droid_visualization.warmup:
                cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

            droid_visualization.ix += 1
            vis.poll_events()
            vis.update_renderer()

    ### create Open3D visualization ###
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_animation_callback(animation_callback)
    vis.register_key_callback(ord("S"), increase_filter)
    vis.register_key_callback(ord("A"), decrease_filter)
    # no conflicts in key "O" and "E"
    vis.register_key_callback(ord("O"), set_obj_pose_mode)
    vis.register_key_callback(ord("E"), export_scene_points)
    vis.create_window(height=540, width=960)
    vis.get_render_option().load_from_json("misc/renderoption.json")
    print(vis.get_render_option().line_width)
    # exit(0)
    cam = vis.get_view_control().convert_to_pinhole_camera_parameters()

    cam.extrinsic = np.array(
        [
            [0.86405783, -0.12739846, 0.48700483, -0.3105393],
            [-0.02629469, 0.95470218, 0.29639895, 0.11203991],
            [-0.50270534, -0.26891148, 0.82156799, 1.10067384],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

    vis.run()
    vis.destroy_window()
