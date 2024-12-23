import argparse
import glob
import json
import os
import re
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
from torchvision import transforms

def sort_key(path):  
    # 假设文件名总是以 'rgb_' 开头，后跟数字  
    # 使用分割找到最后一个 '/' 后面的部分，再找到 '_' 后面的数字  
    filename = path.split('/')[-1]  
    number_str = filename.split('_')[-1].split('.')[0]  # 去除 .jpg 扩展名  
    return int(number_str)  

def read_rt_matrix(filename):
    """
    从文件中读取相机位姿数据，并转换为旋转矩阵和位移向量。
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
        # if len(lines) != 3:
        #     raise ValueError("位姿文件应包含三行数据。")

        # 读取旋转矩阵和平移向量
        r1 = list(map(float, lines[0].strip().split()))
        r2 = list(map(float, lines[1].strip().split()))
        r3 = list(map(float, lines[2].strip().split()))
        r4 = list([0, 0, 0, 1])

        # 构建旋转矩阵 R 和位移向量 T
        RT = np.array([r1, r2, r3, r4])

        return RT
def alphanum_key(s):
    """Turn a string into a list of string and number chunks.
    "z23a" -> ["z", 23, "a"]
    """
    return [int(x) if x.isdigit() else x for x in re.split("([0-9]+)", s)]


def load_poses(posefile):
    file = open(posefile, "r")
    lines = file.readlines()
    file.close()
    poses = []
    valid = []
    lines_per_matrix = 4
    for i in range(0, len(lines), lines_per_matrix):
        if "nan" in lines[i]:
            valid.append(False)
            poses.append(np.eye(4, 4, dtype=np.float32).tolist())
        else:
            valid.append(True)
            pose_floats = [[float(x) for x in line.split()] for line in lines[i : i + lines_per_matrix]]
            poses.append(pose_floats)

    return poses, valid

# type : choices=["mono_prior", "sensor_depth"]
def data_style_deal(input_path,output_path,type="sensor_depth"):
    output_path = Path(output_path)  # "data/custom/scannet_scene0050_00"
    input_path = Path(input_path)  # "/home/yuzh/Projects/datasets/scannet/scene0050_00"

    output_path.mkdir(parents=True, exist_ok=True)

    # load color
    color_path = os.path.join(input_path, 'rgb')
    color_paths = sorted([color_path + "/" + f for f in os.listdir(color_path) if f.endswith('.png') or f.endswith('.jpg')])
        
    # load depth
    depth_path = os.path.join(input_path, 'depth')
    depth_paths = sorted([depth_path + "/" + f for f in os.listdir(depth_path) if f.endswith('.png')])


    # load intrinsic
    intrinsic_path = input_path / "cam_0.txt"
    camera_intrinsic = np.loadtxt(intrinsic_path)

    # load pose
    poses = []
    pose_path = os.path.join(input_path, 'pose_1')
    pose_paths = sorted([pose_path + "/" + f for f in os.listdir(pose_path) if f.endswith('.txt')])
    for pose_path in pose_paths:
        c2w = read_rt_matrix(pose_path)
        # print(c2w)
        poses.append(c2w)
    # output_path = Path(args.output_path)  # "data/neural_rgbd/breakfast_room/"
    # input_path = Path(args.input_path)  # "data/neural_rgbd_data/breakfast_room/"

    # output_path.mkdir(parents=True, exist_ok=True)

    # # load color
    # color_path = input_path / "images"
    # color_paths = sorted(glob.glob(os.path.join(color_path, "*.png")), key=alphanum_key)

    # # load depth
    # depth_path = input_path / "depth_filtered"
    # depth_paths = sorted(glob.glob(os.path.join(depth_path, "*.png")), key=alphanum_key)

    H, W = cv2.imread(depth_paths[0]).shape[:2]
    print(H, W)

    # # load intrinsic
    # intrinsic_path = input_path / "focal.txt"
    # focal_length = np.loadtxt(intrinsic_path)

    # camera_intrinsic = np.eye(4)
    # camera_intrinsic[0, 0] = focal_length
    # camera_intrinsic[1, 1] = focal_length
    # camera_intrinsic[0, 2] = W * 0.5
    # camera_intrinsic[1, 2] = H * 0.5

    # print(camera_intrinsic)
    # # load pose

    # pose_path = input_path / "poses.txt"
    # poses, valid_poses = load_poses(pose_path)
    poses = np.array(poses)
    valid_poses = np.isfinite(poses).all(axis=2).all(axis=1)
    # print(poses.shape)

    # OpenGL/Blender convention, needs to change to COLMAP/OpenCV convention
    # https://docs.nerf.studio/en/latest/quickstart/data_conventions.html
    # poses[:, 0:3, 1:3] *= -1
    # poses[:, 0:3, 3]/= 1000.0

    # deal with invalid poses
    min_vertices = poses[:, :3, 3][valid_poses].min(axis=0)
    max_vertices = poses[:, :3, 3][valid_poses].max(axis=0)

    center = (min_vertices + max_vertices) / 2.0
    scale = 2.0 / (np.max(max_vertices - min_vertices) + 3.0)
    print(center, scale)

    # we should normalize pose to unit cube
    poses[:, :3, 3] -= center
    poses[:, :3, 3] *= scale

    # inverse normalization
    scale_mat = np.eye(4).astype(np.float32)
    scale_mat[:3, 3] -= center
    scale_mat[:3] *= scale
    scale_mat = np.linalg.inv(scale_mat)

    if type == "mono_prior":
        # center copy image if use monocular prior because omnidata use 384x384 as inputs
        # get smallest side to generate square crop
        target_crop = min(H, W)

        target_size = 384
        trans_totensor = transforms.Compose(
            [
                transforms.CenterCrop(target_crop),
                transforms.Resize(target_size, interpolation=PIL.Image.BILINEAR),
            ]
        )

        # center crop by min_dim
        offset_x = (W - target_crop) * 0.5
        offset_y = (H - target_crop) * 0.5

        camera_intrinsic[0, 2] -= offset_x
        camera_intrinsic[1, 2] -= offset_y
        # resize from min_dim x min_dim -> to 384 x 384
        resize_factor = target_size / target_crop
        camera_intrinsic[:2, :] *= resize_factor

        # new H, W after center crop
        H, W = target_size, target_size

    K = camera_intrinsic

    frames = []
    out_index = 0
    for idx, (valid, pose, image_path, depth_path) in enumerate(zip(valid_poses, poses, color_paths, depth_paths)):

        # if idx % 10 != 0:
        #     continue
        # if not valid:
            
        # number = continuesort_key(image_path)
        # if idx%2==0:
        #     continue

        target_image = output_path / f"{out_index:06d}_rgb.png"
        print(target_image)
        if type == "mono_prior":
            img = Image.open(image_path)
            img_tensor = trans_totensor(img)
            img_tensor.save(target_image)
        else:
            shutil.copyfile(image_path, target_image)

        rgb_path = str(target_image.relative_to(output_path))
        frame = {
            "rgb_path": rgb_path,
            "camtoworld": pose.tolist(),
            "intrinsics": K.tolist(),
        }
        if type == "mono_prior":
            frame.update(
                {
                    "mono_depth_path": rgb_path.replace("_rgb.png", "_depth.npy"),
                    "mono_normal_path": rgb_path.replace("_rgb.png", "_normal.npy"),
                }
            )
        else:
            frame["sensor_depth_path"] = rgb_path.replace("_rgb.png", "_depth.npy")

            depth_map = cv2.imread(depth_path, -1)
            # Convert depth to meters, then to "network units"
            depth_shift = 21.845
            depth_maps = (np.array(depth_map) / depth_shift).astype(np.float32)
            # print(depth_maps.max())
            depth_maps *= scale

            np.save(output_path / frame["sensor_depth_path"], depth_maps)

            # color map gt depth for visualization
            plt.imsave(output_path / frame["sensor_depth_path"].replace(".npy", ".png"), depth_maps, cmap="viridis")

        frames.append(frame)
        out_index += 1

    # scene bbox for the scannet scene
    scene_box = {
        "aabb": [[-1, -1, -1], [1, 1, 1]],
        "near": 0.05,
        "far": 2.5,
        "radius": 1.0,
        "collider_type": "box",
    }

    # meta data
    output_data = {
        "camera_model": "OPENCV",
        "height": H,
        "width": W,
        "has_mono_prior": type == "mono_prior",
        "has_sensor_depth": type == "sensor_depth",
        "pairs": None,
        "worldtogt": scale_mat.tolist(),
        "scene_box": scene_box,
    }

    output_data["frames"] = frames

    # save as json
    with open(output_path / "meta_data.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)
    print(f"Saved {len(frames)} frames to {output_path}")
