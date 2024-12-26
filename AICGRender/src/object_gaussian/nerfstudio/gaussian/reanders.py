""" This module is responsible for evaluating rendering, trajectory and reconstruction metrics"""
import os
import traceback
from argparse import ArgumentParser
from copy import deepcopy
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import trimesh  
import torchvision
from PIL import Image
from scipy.ndimage import median_filter
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...nerfstudio.gaussian.io_utils import load_config
from ...nerfstudio.gaussian.scene.gaussian_model import GaussianModel
from ...nerfstudio.gaussian.datasets import get_dataset
from ...nerfstudio.gaussian.merged_map import (RenderFrames, merge_submaps,
                                                refine_global_map)
from ...nerfstudio.gaussian.util import (get_render_settings, np2torch,
                                      render_gaussian_model, setup_seed, torch2np)


def filter_depth_outliers(depth_map, kernel_size=3, threshold=1.0):
    median_filtered = median_filter(depth_map, size=kernel_size)
    abs_diff = np.abs(depth_map - median_filtered)
    outlier_mask = abs_diff > threshold
    depth_map_filtered = np.where(outlier_mask, median_filtered, depth_map)
    return depth_map_filtered


class Reanders(object):

    def __init__(self, config_path,input_path,save_path,scene_type=0,config=None) -> None:
        if config is None:
            self.config = load_config(config_path)
        else:
            self.config = config
        # output_image_path = save_path # self.config["data"]["output_image_path"]
        os.makedirs(save_path, exist_ok=True)
        self.config["data"]["input_path"]=input_path
        self.config["data"]["scene_type"] = scene_type
        setup_seed(self.config["seed"])
        self.device = "cuda"
        self.dataset = get_dataset()({**self.config["data"], **self.config["cam"]})
        self.dataset.output_image_path = save_path
        self.scene_name = self.config["data"]["scene_name"]
        self.dataset_name = self.config["dataset_name"]
        self.gt_poses = np.array(self.dataset.poses)
        self.fx, self.fy = self.dataset.intrinsics[0, 0], self.dataset.intrinsics[1, 1]
        self.cx, self.cy = self.dataset.intrinsics[0,
        2], self.dataset.intrinsics[1, 2]
        self.width, self.height = self.dataset.width, self.dataset.height

    def run_global_map_create(self, obj_mesh):
        training_frames = RenderFrames(self.dataset, self.gt_poses, self.height, self.width, self.fx, self.fy)
        training_frames = DataLoader(training_frames, batch_size=1, shuffle=True)
        training_frames = cycle(training_frames)
        # ply_dir = os.path.join(self.dataset.dataset_path, "result.obj")
        # # merged_cloud = o3d.io.read_point_cloud(ply_dir)
        # mesh = o3d.io.read_triangle_mesh(ply_dir)
        merged_cloud = o3d.geometry.PointCloud()
        # 将点云数据赋值给点云对象
        merged_cloud.points = o3d.utility.Vector3dVector(obj_mesh.vertices)

        if obj_mesh.vertex_normals is not None:
            normals = np.asarray(obj_mesh.vertex_normals)
            merged_cloud.normals = o3d.utility.Vector3dVector(normals)
            print("normals ",len(normals))

        if obj_mesh.visual.vertex_colors is not None:
            colors = np.asarray(obj_mesh.visual.vertex_colors[:, :3]) / 255.0  # 转换到 [0, 1] 范围
            merged_cloud.colors = o3d.utility.Vector3dVector(colors)
            print("color ",len(colors))

        #------------------------------------------GS模型训练----------------------------------------------#
        refined_merged_gaussian_model = refine_global_map(merged_cloud, training_frames, 50000,self.dataset.dataset_path,self.dataset.output_image_path,self.dataset)

        # self.out_new_rgb(self.dataset.dataset_path,self.dataset.output_image_path, refined_merged_gaussian_model)
        voxel_size = 0.01
        pcd =merged_cloud











        #-----------------------------------------------------------------------------------------------#

























        #训练gs模型

        # pcd = merged_cloud.voxel_down_sample(voxel_size)
        # # 估计法线
        # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # # 执行 Poisson 重建
        # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9, width=0, scale=1,
        #                                                                             linear_fit=True)
        #裁剪多余的边
        # densities = np.asarray(densities)
        # vertices_to_remove = densities < np.quantile(densities, 0.004)
        # mesh.remove_vertices_by_mask(vertices_to_remove)
        # 可选：平滑处理
        # mesh.filter_smooth_simple(number_of_iterations=20)
        # o3d.io.write_triangle_mesh(mesh_out + "result.obj", mesh)
        print('*'*20)
        print('*'*20)

    def out_new_rgb(self, input_, out_put, gaussian_model: GaussianModel):
        dst = self.read_std_rt_matrix(os.path.join(input_, 'readme.txt'))
        # dst = self.dataset.poses
        print()
        for idx in range(len(dst)):
            dst_pose = np.linalg.inv(dst[idx])
            render_settings = get_render_settings(self.dataset.width, self.dataset.height, self.dataset.intrinsics,
                                                  dst_pose)
            render_pkg_vis = render_gaussian_model(gaussian_model, render_settings)
            image_vis = render_pkg_vis["color"]  # Convert tensor to NumPy array
            image_vis = image_vis.clone().detach().permute(1, 2, 0)
            color_np = image_vis.detach().cpu().numpy()
            color_np = np.clip(color_np, 0, 1)
            image_pil = Image.fromarray((color_np * 255).astype(np.uint8))
            image_pil.save(f"{out_put}/rgb_{idx}.jpg")

    def read_std_rt_matrix(self, filename):
        dst = []
        """
            从文件中读取相机位姿数据，并转换为旋转矩阵和位移向量。
            """
        with open(filename, 'r') as file:
            lines = file.readlines()
            if len(lines) != 25:
                raise ValueError("位姿文件应包含三行数据。")
            start_index = 6
            for i in range(5):
                # 读取旋转矩阵和平移向量
                r1 = list(map(float, lines[start_index].strip().split()))
                r2 = list(map(float, lines[start_index + 1].strip().split()))
                r3 = list(map(float, lines[start_index + 2].strip().split()))
                r4 = list([0, 0, 0, 1])
                # 构建旋转矩阵 R 和位移向量 T
                RT = np.array([r1, r2, r3, r4])
                start_index = start_index + 4
                dst.append(RT)
        return dst
