import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import json
import imageio


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_config: dict):
        self.dataset_path = Path(dataset_config["input_path"])
        self.output_mesh_path = Path(dataset_config["output_mesh_path"])
        self.output_image_path = Path(dataset_config["output_image_path"])
        self.frame_limit = dataset_config.get("frame_limit", -1)
        self.dataset_config = dataset_config
        self.height = dataset_config["H"]
        self.width = dataset_config["W"]
        self.fx = dataset_config["fx"]
        self.fy = dataset_config["fy"]
        self.cx = dataset_config["cx"]
        self.cy = dataset_config["cy"]

        self.depth_scale = dataset_config["depth_scale"]
        self.distortion = np.array(
            dataset_config['distortion']) if 'distortion' in dataset_config else None
        self.crop_edge = dataset_config['crop_edge'] if 'crop_edge' in dataset_config else 0
        if self.crop_edge:
            self.height -= 2 * self.crop_edge
            self.width -= 2 * self.crop_edge
            self.cx -= self.crop_edge
            self.cy -= self.crop_edge

        self.fovx = 2 * math.atan(self.width / (2 * self.fx))
        self.fovy = 2 * math.atan(self.height / (2 * self.fy))
        self.intrinsics = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        self.color_paths = []
        self.depth_paths = []

    def __len__(self):
        return len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit)


class My_RGBD(BaseDataset):
  
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        # if(self)
        rgb_dir = os.path.join(self.dataset_path, 'rgb')
        depth_dir = os.path.join(self.dataset_path, 'depth')
        pose_dir = os.path.join(self.dataset_path, 'pose_1')

        self.color_paths = sorted([rgb_dir+"/"+f for f in os.listdir(rgb_dir) if f.endswith('.png') or f.endswith('.jpg')])
        self.depth_paths = sorted([depth_dir+"/"+f for f in os.listdir(depth_dir) if f.endswith('.png')])
        pose_files = sorted([pose_dir+"/"+f for f in os.listdir(pose_dir) if f.endswith('.txt')])

        selected=[i for i in range(0,len(pose_files),2)]
        self.color_paths =[self.color_paths[i] for i in selected]
        self.depth_paths =[self.depth_paths[i] for i in selected]
        pose_files =[pose_files[i] for i in selected]

        self.load_poses(pose_files)
        print(f"Loaded {len(self.color_paths)} frames")
    def read_rt_matrix(self,filename):
        """
        从文件中读取相机位姿数据，并转换为旋转矩阵和位移向量。
        """
        with open(filename, 'r') as file:
            lines = file.readlines()
            if len(lines) < 3:
                raise ValueError("位姿文件应包含三行数据。")

            # 读取旋转矩阵和平移向量
            r1 = list(map(float, lines[0].strip().split()))
            r2 = list(map(float, lines[1].strip().split()))
            r3 = list(map(float, lines[2].strip().split()))
            r4 = list([0,0,0,1])

            # 构建旋转矩阵 R 和位移向量 T
            RT = np.array([r1, r2, r3,r4])
            return RT    
    
    def load_poses(self, paths):
        self.poses = []
        for path in paths:
            c2w = self.read_rt_matrix(path)
            self.poses.append(c2w.astype(np.float32))

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        return index, color_data, depth_data, self.poses[index]

#可扩展
def get_dataset():
        return My_RGBD
