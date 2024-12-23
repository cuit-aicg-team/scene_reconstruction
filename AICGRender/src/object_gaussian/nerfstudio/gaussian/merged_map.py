""" This module is responsible for merging submaps. """
from argparse import ArgumentParser

import faiss
import numpy as np
import os
import open3d as o3d
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as thf
from ...nerfstudio.gaussian.arguments import OptimizationParams
from ...nerfstudio.gaussian.scene.gaussian_model import GaussianModel
from ...nerfstudio.gaussian.losses import isotropic_loss, l1_loss, ssim,scale_normal_loss,psnr
from ...nerfstudio.gaussian.util import (batch_search_faiss, get_render_settings,
                                      np2ptcloud, render_gaussian_model, torch2np)
from PIL import Image
import cv2
from scipy.spatial import cKDTree
from  ...nerfstudio.gaussian.gaussian_renderer import render,render_test
from ...nerfstudio.gaussian.arguments import ModelParams, PipelineParams, OptimizationParams
from random import randint
class RenderFrames(Dataset):
    """A dataset class for loading keyframes along with their estimated camera poses and render settings."""
    def __init__(self, dataset, render_poses: np.ndarray, height: int, width: int, fx: float, fy: float):
        self.dataset = dataset
        self.render_poses = render_poses
        self.height = height
        self.width = width
        self.fx = fx
        self.fy = fy
        self.device = "cpu"
        self.stride = 1
        if len(dataset) > 1000:
            self.stride = len(dataset) // 1000

    def __len__(self) -> int:
        return len(self.dataset) // self.stride

    def __getitem__(self, idx):
        idx = idx * self.stride
        color = (torch.from_numpy(
            self.dataset[idx][1]) / 255.0).float().to(self.device)
        depth = torch.from_numpy(self.dataset[idx][2]).float().to(self.device)
        estimate_c2w = self.render_poses[idx]
        estimate_w2c = np.linalg.inv(estimate_c2w)
        frame = {
            "frame_id": idx,
            "color": color,
            "depth": depth,
            "render_settings": get_render_settings(
                self.width, self.height, self.dataset.intrinsics, estimate_w2c)
        }
        return frame


def merge_submaps(submaps_paths: list, radius: float = 0.0001, device: str = "cuda") -> o3d.geometry.PointCloud:
    """ Merge submaps into a single point cloud, which is then used for global map refinement.
    Args:
        segments_paths (list): Folder path of the submaps.
        radius (float, optional): Nearest neighbor distance threshold for adding a point. Defaults to 0.0001.
        device (str, optional): Defaults to "cuda".

    Returns:
        o3d.geometry.PointCloud: merged point cloud
    """
    pts_index = faiss.IndexFlatL2(3)
    if device == "cuda":
        pts_index = faiss.index_cpu_to_gpu(
            faiss.StandardGpuResources(),
            0,
            faiss.IndexIVFFlat(faiss.IndexFlatL2(3), 3, 500, faiss.METRIC_L2))
        pts_index.nprobe = 5
    merged_pts = []
    print("Merging segments")
    for submap_path in tqdm(submaps_paths):
        gaussian_params = torch.load(submap_path)["gaussian_params"]
        current_pts = gaussian_params["xyz"].to(device).float()
        pts_index.train(current_pts)
        distances, _ = batch_search_faiss(pts_index, current_pts, 8)
        neighbor_num = (distances < radius).sum(axis=1).int()
        ids_to_include = torch.where(neighbor_num == 0)[0]
        pts_index.add(current_pts[ids_to_include])
        merged_pts.append(current_pts[ids_to_include])
    pts = torch2np(torch.vstack(merged_pts))
    pt_cloud = np2ptcloud(pts, np.zeros_like(pts))

    # Downsampling if the total number of points is too large
    if len(pt_cloud.points) > 1_000_000:
        voxel_size = 0.04
        pt_cloud = pt_cloud.voxel_down_sample(voxel_size)
        print(f"Downsampled point cloud to {len(pt_cloud.points)} points")
    filtered_pt_cloud, _ = pt_cloud.remove_statistical_outlier(nb_neighbors=40, std_ratio=3.0)
    del pts_index
    return filtered_pt_cloud


def read_std_rt_matrix(filename):
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


def read_pose_matrix(file_path):
    try:
        pose_matrix = np.loadtxt(file_path)
        ext_matrix = np.zeros((4, 4))
        # 将前三行填入外参矩阵
        ext_matrix[:3, :] = pose_matrix[:3, :]  # 假设前三行是旋转部分
        ext_matrix[3, 3] = 1  # 最后一行设置为 [0, 0, 0, 1]
        return ext_matrix
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import torch.nn.functional as F
def out_new_rgb(dataset, input_, out_put, gaussian_model, iteration,pipe,background):
    dst = read_std_rt_matrix(os.path.join(input_, 'readme.txt'))
    # print()
    for idx in range(len(dst)):
        dst_pose = np.linalg.inv(dst[idx])
        render_settings = get_render_settings(dataset.width, dataset.height, dataset.intrinsics, dst_pose)
        render_pkg = render_test(render_settings, gaussian_model, pipe, background)
        image_vis = render_pkg["render"].clone().detach().permute(1, 2, 0)
        color_np = image_vis.detach().cpu().numpy()
        color_np = np.clip(color_np, 0, 1)
        image_pil = Image.fromarray((color_np * 255).astype(np.uint8))
        image_pil.save(f"{out_put}/rgb_{idx}.jpg")

    #########测试test图片指标#######
    # 创建指标计算对象
    ssim_metric = StructuralSimilarityIndexMeasure()
    psnr_metric = PeakSignalNoiseRatio()

    test_path = os.path.join(input_, 'test')
    txt_files = [f for f in os.listdir(test_path) if f.endswith('.txt') and f != 'metrics.txt' and f != 'readme.txt']

    # 记录指标的文件
    metrics_file_path = os.path.join(input_, 'test', 'metrics.txt')


    # 确保文件存在并写入表头
    if(iteration==500):
        with open(metrics_file_path, 'w') as metrics_file:
            metrics_file.write("Iteration\tPSNR\tSSIM\tL1Loss\n")

    psnr_total = 0
    ssim_total = 0
    l1_loss_total = 0
    for idx, pose_file in enumerate(txt_files):
        pose = np.linalg.inv(read_pose_matrix(os.path.join(test_path, pose_file)))
        render_settings = get_render_settings(dataset.width, dataset.height, dataset.intrinsics, pose)
        render_pkg = render_test(render_settings, gaussian_model, pipe, background)
        image_vis = render_pkg["render"]  # image_vis维度为(3,1080,1080)

        image_vis = image_vis.clone().detach().permute(1, 2, 0).cpu().numpy()
        image_vis = np.clip(image_vis, 0, 1)
        image_vis = image_vis[..., [2, 1, 0]]  # 交换红色和蓝色通道


        rgb_file_name = pose_file.replace('RT', 'rgb').replace('.txt', '.jpg')
        rgb_file_path = os.path.join(input_, 'test', rgb_file_name)
        gt_color = cv2.imread(rgb_file_path)
        # gt_color = cv2.cvtColor(gt_color, cv2.COLOR_BGR2RGB)
        gt_color = gt_color / 255.0  # 归一化

        # 计算PSNR
        psnr_value = psnr_metric(torch.tensor(image_vis).permute(2, 0, 1).unsqueeze(0), torch.tensor(gt_color).permute(2, 0, 1).unsqueeze(0))
        psnr_total += psnr_value.item()

        # 计算SSIM
        ssim_value = ssim_metric(torch.tensor(image_vis).permute(2, 0, 1).unsqueeze(0), torch.tensor(gt_color).permute(2, 0, 1).unsqueeze(0))
        ssim_total += ssim_value.item()

        # 计算L1 Loss
        l1_loss_value = F.l1_loss(torch.tensor(image_vis), torch.tensor(gt_color))
        l1_loss_total += l1_loss_value.item()

        # 保存image_vis
        image_vis_uint8 = (image_vis * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_put, f'test_{idx}.jpg'), image_vis_uint8)

        print(f"iteration:{iteration}:{idx}\t{psnr_value:.4f}\t{ssim_value:.4f}\t{l1_loss_value:.4f}\n")

    # 记录每次iteration的平均指标
    avg_psnr = psnr_total / len(txt_files)
    avg_ssim = ssim_total / len(txt_files)
    avg_l1_loss = l1_loss_total / len(txt_files)

    with open(metrics_file_path, 'a') as metrics_file:
        metrics_file.write(f"iteration:{iteration}\t{avg_psnr:.4f}\t{avg_ssim:.4f}\t{avg_l1_loss:.4f}\n")


def downsample_point_cloud(pt_cloud, target_num_points=10000):
    """
    Downsample a point cloud to a specified number of points, preserving colors.

    Parameters:
    - pt_cloud (o3d.geometry.PointCloud): The input point cloud.
    - target_num_points (int): The desired number of points after downsampling.

    Returns:
    - o3d.geometry.PointCloud: The downsampled point cloud with preserved colors.
    """
    num_points = np.asarray(pt_cloud.points).shape[0]

    if num_points <= target_num_points:
        return pt_cloud

    # Randomly select indices for downsampling
    indices = np.random.choice(num_points, size=target_num_points, replace=False)

    # Downsample points and colors
    downsampled_points = np.asarray(pt_cloud.points)[indices]
    if pt_cloud.has_colors():
        downsampled_colors = np.asarray(pt_cloud.colors)[indices]
    else:
        downsampled_colors = None

    # Create a new point cloud
    downsampled_cloud = o3d.geometry.PointCloud()
    downsampled_cloud.points = o3d.utility.Vector3dVector(downsampled_points)
    if downsampled_colors is not None:
        downsampled_cloud.colors = o3d.utility.Vector3dVector(downsampled_colors)

    return downsampled_cloud



from argparse import ArgumentParser, Namespace
import sys
def get_combined_args(parser: ArgumentParser):
    # 解析命令行参数
    cmdlne_string = sys.argv[1:]
    args_cmdline = parser.parse_args(cmdlne_string)

    # 读取配置文件参数
    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    # 合并来自命令行参数和配置文件的参数
    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v

    # 返回合并后的参数
    return Namespace(**merged_dict)
def refine_global_map(pt_cloud: o3d.geometry.PointCloud, training_frames: list, max_iterations: int,input,output,dataset) -> GaussianModel:
    """Refines a global map based on the merged point cloud and training keyframes frames.
    Args:
        pt_cloud (o3d.geometry.PointCloud): The merged point cloud used for refinement.
        training_frames (list): A list of training frames for map refinement.
        max_iterations (int): The maximum number of iterations to perform for refinement.
    Returns:
        GaussianModel: The refined global map as a Gaussian model.
    """
    first_iter = 0
    #训练参数
    lp = ModelParams()
    opt = OptimizationParams()
    pipe = PipelineParams()


    gaussians = GaussianModel(3)
    # gaussian_model.active_sh_degree = 3
    gaussians.create_from_pcd(downsample_point_cloud(pt_cloud,50000),1.0)
    gaussians.training_setup(opt)
    checkpoint =False
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if lp.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    with torch.enable_grad():
        for iteration in range(first_iter, opt.iterations + 1):
            if iteration % 500 == 0:
                outpath = output.joinpath(iteration.__str__())
                os.makedirs(outpath,exist_ok=True)
                out_new_rgb(dataset, input, outpath, gaussians,iteration,pipe,background)

            iter_start.record()
            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()


            training_frame = next(training_frames)

            gt_color, gt_depth, render_settings = (
                training_frame["color"].squeeze(0),
                training_frame["depth"].squeeze(0),
                training_frame["render_settings"])

            render_pkg = render(render_settings,gaussians, pipe, background)

            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]

            gt_image = gt_color.permute(2, 0, 1).to("cuda")
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            # regularization
            lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
            lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

            rend_dist = render_pkg["rend_dist"]
            rend_normal = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()
            dist_loss = lambda_dist * (rend_dist).mean()

            # loss
            total_loss = loss + dist_loss + normal_loss

            total_loss.backward()

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
                ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

                if iteration % 10 == 0:
                    loss_dict = {
                        "Loss": f"{ema_loss_for_log:.{5}f}",
                        "distort": f"{ema_dist_for_log:.{5}f}",
                        "normal": f"{ema_normal_for_log:.{5}f}",
                        "Points": f"{len(gaussians.get_xyz)}"
                    }
                    progress_bar.set_postfix(loss_dict)

                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Densification
                if iteration < opt.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                         radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        # gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent,
                        #                             size_threshold)
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, 2.0,size_threshold)
                    if iteration % opt.opacity_reset_interval == 0 or (
                            lp.white_background and iteration == opt.densify_from_iter):
                        pass
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                if (iteration in [7000,30000,50000]):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    # torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    return gaussians



################################################################################################################################
    # pbar = tqdm(range(1, max_iterations+1))
    # for iteration in pbar:
    #     with torch.enable_grad():
    #         if (iteration+1) % 500 == 0:
    #             outpath = output.joinpath(iteration.__str__())
    #             os.makedirs(outpath,exist_ok=True)
    #             out_new_rgb(dataset, input, outpath, gaussian_model,iteration)
    #
    #
    #         training_frame = next(training_frames)
    #         gt_color, gt_depth, render_settings = (
    #             training_frame["color"].squeeze(0),
    #             training_frame["depth"].squeeze(0),
    #             training_frame["render_settings"])
    #
    #         render_dict = render_gaussian_model(gaussian_model, render_settings)
    #         rendered_color, rendered_depth = (render_dict["color"].permute(1, 2, 0), render_dict["depth"].squeeze(0))
    #
    #         #计算损失
    #         reg_loss = isotropic_loss(gaussian_model.get_scaling())
    #         depth_mask = (gt_depth > 0)
    #         depth_mask_expanded = depth_mask.unsqueeze(-1).expand_as(gt_color)
    #
    #         zero_render_color = torch.zeros_like(rendered_color)
    #         color_loss = (1.0 - opt_params.lambda_dssim) * l1_loss(
    #             rendered_color[depth_mask, :], gt_color[depth_mask, :]
    #         ) + opt_params.lambda_dssim * (1.0 - ssim(rendered_color, torch.where(depth_mask_expanded, gt_color, zero_render_color)))
    #
    #
    #         depth_loss = l1_loss(rendered_depth[depth_mask], gt_depth[depth_mask])
    #
    #         Ll2 = thf.mse_loss(rendered_color[depth_mask, :], gt_color[depth_mask, :])
    #
    #
    #         total_loss = color_loss + reg_loss + Ll2
    #         total_loss.backward()
    #
    #         psnr_value = psnr(rendered_color[depth_mask, :], gt_color[depth_mask, :]).mean().float().item()
    #         pbar.set_postfix({
    #             'loss': total_loss.item(),
    #             'psnr': psnr_value,
    #             'gauss': gaussian_model.get_xyz().shape[0],
    #         })
    #
    #         gaussian_model.adaptive_density_control(render_dict, iteration)
    #
    #     with torch.no_grad():
    #         # Optimizer step
    #         gaussian_model.optimizer.step()
    #         gaussian_model.optimizer.zero_grad(set_to_none=True)
    #

