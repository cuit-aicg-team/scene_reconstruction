import argparse
import glob
import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
from torchvision import transforms



# copy image
H, W = 2048, 2448
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
parser = argparse.ArgumentParser(description="preprocess scannet dataset to sdfstudio dataset")

parser.add_argument("--input_path", dest="input_path", help="path to scannet scene")
parser.set_defaults(im_name="NONE")

parser.add_argument("--output_path", dest="output_path", help="path to output")
parser.set_defaults(store_name="NONE")
parser.add_argument(
    "--type",
    dest="type",
    default="mono_prior",
    choices=["mono_prior", "sensor_depth"],
    help="mono_prior to use monocular prior, sensor_depth to use depth captured with a depth sensor (gt depth)",
)

args = parser.parse_args()

# image_size = 1024
trans_totensor = transforms.Compose(
    [
        # transforms.CenterCrop(image_size * 2),
        transforms.Resize([H, W], interpolation=PIL.Image.NEAREST),
    ]
)

depth_trans_totensor = transforms.Compose(
    [
        transforms.Resize([H, W], interpolation=PIL.Image.NEAREST),
        # transforms.CenterCrop(image_size * 2),
        # transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
    ]
)

output_path = Path(args.output_path)  # "data/custom/scannet_scene0050_00"
input_path = Path(args.input_path)  # "/home/yuzh/Projects/datasets/scannet/scene0050_00"

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
pose_path = os.path.join(input_path, 'pose_1')
poses = []
pose_paths = sorted([pose_path + "/" + f for f in os.listdir(pose_path) if f.endswith('.txt')])
for pose_path in pose_paths:
    c2w = read_rt_matrix(pose_path)
    # print(c2w)
    poses.append(c2w)
poses = np.array(poses)
poses[:, :3, 3] /= 1000.0
poses[:, 0:3, 1:3] *= -1

# deal with invalid poses
valid_poses = np.isfinite(poses).all(axis=2).all(axis=1)
min_vertices = poses[:, :3, 3][valid_poses].min(axis=0)
max_vertices = poses[:, :3, 3][valid_poses].max(axis=0)

center = (min_vertices + max_vertices) / 2.0
scale = 2.0 / (np.max(max_vertices - min_vertices) + 3.0)
# center = 0
# scale =1
print(center, scale)

# we should normalize pose to unit cube
poses[:, :3, 3] -= center
poses[:, :3, 3] *= scale

# inverse normalization
scale_mat = np.eye(4).astype(np.float32)
scale_mat[:3, 3] -= center
scale_mat[:3] *= scale
scale_mat = np.linalg.inv(scale_mat)


# center crop by 2 * image_size
offset_x = 0
offset_y = 0
camera_intrinsic[0, 2] -= offset_x
camera_intrinsic[1, 2] -= offset_y
# resize from 384*2 to 384
resize_factor = 1
camera_intrinsic[:2, :] *= resize_factor

K = camera_intrinsic

frames = []
out_index = 0
for idx, (pose, image_path, depth_path) in enumerate(zip(poses, color_paths, depth_paths)):

    # if idx % 10 != 0:
    #     continue
    # print(len(pose_paths),len(depth_paths),len(color_paths))
    # if not valid:
    #     continue

    
    target_image = output_path / f"{out_index:06d}_rgb.png"
    print(target_image)
    img = Image.open(image_path)
    img_tensor = trans_totensor(img)
    img_tensor.save(target_image)

    # load depth
    target_depth_image = output_path / f"{out_index:06d}_depth.png"
    depth = cv2.imread(depth_path, -1).astype(np.float32)
    depth /= 21845.0
    print("s_depth",depth.max())
  
    depth_PIL = Image.fromarray(depth)
    new_depth = depth_trans_totensor(depth_PIL)
    new_depth = np.asarray(new_depth).copy()
    # scale depth as we normalize the scene to unit box
    new_depth *= scale
    print("depth",new_depth.max(),pose.tolist())
    plt.imsave(target_depth_image, new_depth, cmap="viridis")
    np.save(str(target_depth_image).replace(".png", ".npy"), new_depth)

    rgb_path = str(target_image.relative_to(output_path))
    frame = {
        "rgb_path": rgb_path,
        "camtoworld": pose.tolist(),
        "intrinsics": K.tolist(),
        "mono_depth_path": rgb_path.replace("_rgb.png", "_depth.npy"),
        "mono_normal_path": rgb_path.replace("_rgb.png", "_normal.npy"),
        "sensor_depth_path": rgb_path.replace("_rgb.png", "_depth.npy"),
    }
    # if idx >2:
    #     break

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
    "has_mono_prior": True,
    "has_sensor_depth": True,
    "pairs": None,
    "worldtogt":  scale_mat.tolist(),
    "scene_box": scene_box,
}

output_data["frames"] = frames

# save as json
with open(output_path / "meta_data.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4)