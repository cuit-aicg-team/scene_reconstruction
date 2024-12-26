#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from .scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import sys
import torchvision
from .utils.general_utils import safe_state
from argparse import ArgumentParser
# from .arguments import ModelParams, PipelineParams, get_combined_args
from .scene import GaussianModel,render
from .utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from .utils.render_utils import generate_path, create_videos



class ModelParams(): 
    def __init__(self):
        self.sh_degree = 3
        self.source_path = ""
        self.model_path = ""
        self.images = "images"
        self.resolution = -1
        self.iteration = -1
        self.skip_test = False
        self.render_path = False
        self.white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']

class PipelineParams():
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0
        self.debug = False


class OptimizationParams():
    def __init__(self):
        self.iterations = 50
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0
        self.lambda_normal = 0.05
        self.opacity_cull = 0.05

        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002




def run(colmap_path,gaussian_model_path,save_path):

    # Set up command line argument parser
    # parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams()
    pipeline = PipelineParams()
    # parser.add_argument("--iteration",type=int, default=3000)
    # parser.add_argument('images', type=str, help='Path to the images')
    # args = get_combined_args(parser)
    model.model_path = gaussian_model_path
    model.source_path = colmap_path
    print("Rendering " + model.model_path)
 # -s colmap_path -m 
    dataset, iteration, pipe = model, model.iteration, pipeline
    print("dafsf -                  ----------------------------------------- ",dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_dir = os.path.join(model.model_path, 'test_new_view_render', "ours_{}".format(scene.loaded_iter))
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)

    if (not model.skip_test) and (len(scene.getTrainCameras()) > 0):
        print("export rendered testing images ...")
        os.makedirs(test_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTrainCameras())
        gaussExtractor.export_image(test_dir)

    if model.render_path:
        print("render videos ...")
        traj_dir = os.path.join(model.model_path, 'traj', "ours_{}".format(scene.loaded_iter))
        os.makedirs(traj_dir, exist_ok=True)
        n_fames = 240
        cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
        gaussExtractor.reconstruction(cam_traj)
        gaussExtractor.export_image(traj_dir)
        create_videos(base_dir=traj_dir,
                      input_dir=traj_dir,
                      out_name='render_traj',
                      num_frames=n_fames)

