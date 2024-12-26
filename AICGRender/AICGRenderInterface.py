# Copyright (C) 2024, Inria
# AICG_TEAM research group
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact xxxx
import AICGRender.src._Convert as _Convert
from AICGRender.src.depth_anything_v2._dpt import DepthAnythingV2
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import os
import requests
from tqdm import tqdm
import shutil

import AICGRender.src.outdoor_gaussian.train as OutdoorReconstruction
import AICGRender.src.outdoor_gaussian.render_new_view as OutdoorNewViewRender

import AICGRender.src.object_gaussian.scripts.train as ObjectSReconstruction
import AICGRender.src.object_gaussian.scripts.extract_mesh as ExtractMesh
import AICGRender.src.object_gaussian.scripts.texture as ExtractTextureMesh

import AICGRender.src.object_gaussian.scripts.datasets.process_neuralrgbd_to_sdfstudio_game as NeuralRGBD
import re
# import AICGRender.src.indoor_gaussian.slam as IndoorReconstruction
depth_pth_url="https://cdn-lfs-us-1.hf.co/repos/ef/a0/efa040f8dfeabb0d7e03dde47070ea9d72db9ffe066eb5d6a44a7a2803a1477c/a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27depth_anything_v2_vitl.pth%3B+filename%3D%22depth_anything_v2_vitl.pth%22%3B&Expires=1734697504&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTczNDY5NzUwNH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmhmLmNvL3JlcG9zL2VmL2EwL2VmYTA0MGY4ZGZlYWJiMGQ3ZTAzZGRlNDcwNzBlYTlkNzJkYjlmZmUwNjZlYjVkNmE0NGE3YTI4MDNhMTQ3N2MvYTdlYTE5ZmEwZWQ5OTI0NGU2N2I2MjRjNzJiODU4MGI3ZTk1NTMwNDMyNDU5MDViZTU4Nzk2YTYwOGViOTM0NT9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoifV19&Signature=Eix5%7Ex0ROIT6-ZAJ9lHudgxt7TGbg7E3DraidswYYuVrrxIJH6oaqabk%7EPU%7ElmqUu5efkTamgazeckoA2wdKYR8wkRCiZh8Hxt7eiwR-zfES3Wi3O-oqhA7OPi9xApvs2GzHUI6fSdiCNQEpEeAqouI6WnFEPHzs0vPGQthmlFWVnYq3egnVh1jfhCHs44LQxR6c3q89iUooEqqWKnF6oUuqwujFSXQZ9W2e2Byj1If-77067tKF0l43ZWFcEylqKSxoTrOYORJVgEW6yMq1ioDxoK5xKzQCAYSoLuS-91t4EHlm0XBFjT0z1-BJoLrM00Sy-znzsp49sxoezlCbkA__&Key-Pair-Id=K24J24Z295AEI9"  
default_pth_download_path="./download"
default_point_out_path="./output/points"
default_depth_out_path='./output/depth'
default_outdoor_model_out_path="./output/outdoor/model"
default_outdoor_render_out_path="./output/outdoor/render"
default_indoor_model_out_path="./output/indoor/model"
default_indoor_render_out_path="./output/indoor/render"
default_object_model_out_path="./output/objectdoor/model"
default_objeect_render_out_path="./output/objectdoor/render"
default_deal_style_out_path="./output/dealStyle"
default_model_out_path="./output/dealStyle/out"

def extract_number(filename):   
    return [int(x) if x.isdigit() else x for x in re.split("([0-9]+)", filename)]

def gamma_correction(img, gamma=2.2):
    """
    对图像应用伽马校正（用于8位图像）。
    """
    inv_gamma = 1.0 / gamma
    lookup_table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, lookup_table)

def rgb_to_grayscale(img):
    """
    将RGB图像转换为灰度图像
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def convert_8bit_rgb_to_16bit_grayscale(img, gamma=2.2):
    """
    将8位RGB图像转换为16位灰度图像。
    """
    # 对输入图像进行伽马校正
    img_corrected = gamma_correction(img, gamma)
    
    # 将RGB图像转换为灰度图像
    gray_img = rgb_to_grayscale(img_corrected)
    
    # 将灰度图像从8位转换为16位
    gray_img_16bit = np.uint16(gray_img) * 257  # 8位到16位的转换
    
    return gray_img_16bit

class ReconstructData:
    def __init__(self, rgb_images=None, depth_images=None, camera_extrinsics=None, camera_intrinsics=None, point_cloud=None,other_param=None,reconstruct_model=None,depth_scale =21.845,**kwargs):
        self.__rgb_images = rgb_images
        self.__depth_images = depth_images
        self.__depth_scale = depth_scale
        self.__camera_extrinsics = camera_extrinsics
        self.__camera_intrinsics = camera_intrinsics
        self.__point_cloud = point_cloud
        self.__reconstruct_model = reconstruct_model
        self.__params = kwargs
    def set_rgb_images(self, rgb_images):
        self.__rgb_images = rgb_images

    def set_depth_images(self, depth_images):
        self.__depth_images = depth_images
    #  __camera_extrinsics
    #  list(
    #     id=1, qvec=array([ 0.99484803,  0.00676536, -0.10065007, -0.01005951]), tvec=array([ 0.1040127 , -0.33116729,  2.91096878])
    #      ...
    #     id=n, qvec=array([ 0.99484803,  0.00676536, -0.10065007, -0.01005951]), tvec=array([ 0.1040127 , -0.33116729,  2.91096878])
    #   )
    def set_camera_extrinsics(self, camera_extrinsics):
        self.__camera_extrinsics = camera_extrinsics
    ###
    # cam_intrinsics {'id': 1, 
    #                 'model': 'PINHOLE',
    #                 'width': 1054, 
    #                 'height': 1872, 
    #                 'params': array([1395.47764379, 1391.04857342,  527.        ,  936.        ]), 
    #                 'distorted': False, 
    #                 'k1': 0.0, 
    #                 'k2': 0.0, 
    #                 'p1': 0.0, 
    #                 'p2': 0.0, 
    #                 'k3': 0.0, 
    #                 'depth_scale': 1.0} 
    ###
    def set_camera_intrinsics(self, camera_intrinsics):
        self.__camera_intrinsics = camera_intrinsics

    def set_point_cloud(self, point_cloud):
        self.__point_cloud = point_cloud

    def set_reconstruct_model(self, reconstruct_model):
        self.__reconstruct_model = reconstruct_model

    def set_depth_scale(self,depth_scale):
        self.__depth_scale = depth_scale

    def set_param(self, key, value):
        self.__params[key] = value

    def get_rgb_images(self):
        return self.__rgb_images

    def get_depth_images(self):
        return self.__depth_images

    def get_camera_extrinsics(self):
        return self.__camera_extrinsics

    def get_camera_intrinsics(self):
        return self.__camera_intrinsics

    def get_point_cloud(self):
        return self.__point_cloud

    def get_reconstruct_model(self):
        return self.__reconstruct_model

    def get_param(self, key):
        return self.__params.get(key)

    def get_depth_scale(self):
        return self.__depth_scale

    def __len__(self):
        return len(self.__rgb_images)


class PointCreator:
    '''
    A Python API class for data processing.
    This class provides a systematic process to:
    1. Load RGB images from a directory.
    2. Load depth images from a directory.
    3. Create a 3D point cloud from RGB images.
    4. Create a 3D point cloud from ReconstructData object. (optional)
    5. Save 3D point data.
    '''
    def __init__(self,config=None):
        '''
        Input:
            config: Currently Configuration file no context. this file is enabled to expand other parameters in future.
        '''
        self.__config = config
        self.outdir = None
        self.colmapConverter =_Convert.ColmapConverter()
        pass

    def aicg_point_create(self, rgb_images_in=None, save_output_path=None, recon_data: ReconstructData = None):
        '''
        Create a 3D point cloud from a set of RGB images
        Input:
            rgb_images_in: List of RGB images
            save_output_path: Path to save the point cloud
            recon_data: ReconstructData object containing RGB and depth images (optional)
        return:
            ReconstructData: ReconstructData object containing RGB images, camera extrinsics, camera intrinsics, and point cloud
        '''
        if rgb_images_in is None:
            raise ValueError("RGB images must be provided")
        
        if save_output_path is None:
            save_output_path = default_point_out_path
            if os.path.exists(default_point_out_path):
                 shutil.rmtree(default_point_out_path)

        args = {"source_path":rgb_images_in,"save_path":save_output_path}
        self.outdir = save_output_path
        self.colmapConverter =_Convert.ColmapConverter(args)
        self.colmapConverter.convert()
        pass


class DepthCreator:
    '''
    A Python API class for data processing.
    This class provides a systematic process to:
    1. Load RGB images from a directory.
    2. Get Depth data from RGB images.
    3. Create a depth map from ReconstructData object. (optional)
    4. Save depth data.
    '''
    def __init__(self,config=None):
        '''
        Input:
            config: Currently Configuration file no context. this file is enabled to expand other parameters in future.
        '''
        self.__config = config
        self.choices=['vits', 'vitb', 'vitl', 'vitg']
        self.img_path =""
        self.input_size=518
        self.outdir= None
        self.encoder=self.choices[2]
        self.pred_only=True  #only display the prediction
        self.grayscale=True  #do not apply colorful palette
        self.depth_anything = None
        
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

    
    def download_file(self,url, save_path, callback=None):
        """
        下载文件到指定路径，并在下载完成后调用回调函数。

        :param url: 目标文件的下载 URL
        :param save_path: 文件保存路径
        :param callback: 下载完成后的回调函数
        """
        try:
            print(f"Downloading Model...")
            response = requests.get(url, stream=True)
            response.raise_for_status()  # 确保请求成功

            # 获取文件总大小
            total_size = int(response.headers.get('Content-Length', 0))
            with open(save_path, 'wb') as file, tqdm(
                desc=save_path,
                total=total_size,
                unit='B',
                unit_scale=True,
                ncols=100
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
                        bar.update(len(chunk))  # 更新进度条

            print(f"File download completed.")

            # 下载完成后，重命名文件去掉 .downloading 后缀
            os.rename(save_path, save_path.replace('.downloading', ''))
            final_save_path = save_path.replace('.downloading', '')
           
            # 下载完成后，调用回调函数
            if callback:
                callback(final_save_path)

        except requests.RequestException as e:
            print(f"Download failed: {e}")

    def on_download_complete(self,file_path):
        """
        下载完成后的回调函数
        :param file_path: 下载完成的文件路径
        """
        self.depth_anything = DepthAnythingV2(**self.model_configs[self.encoder])
        self.depth_anything.load_state_dict(torch.load(file_path, map_location='cpu'))
        self.depth_anything = self.depth_anything.to(self.DEVICE).eval()
        
        if os.path.isfile(self.img_path):
            if self.img_path.endswith('txt'):
                with open(self.img_path, 'r') as f:
                    filenames = f.read().splitlines()
            else:
                filenames = [self.img_path]
        else:
            filenames = glob.glob(os.path.join(self.img_path, '**/*'), recursive=True)
        
        os.makedirs(self.outdir, exist_ok=True)
        
        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        
        for k, filename in enumerate(filenames):
            print(f'Progress {k+1}/{len(filenames)}: {filename}')
            
            raw_image = cv2.imread(filename)
            
            depth = self.depth_anything.infer_image(raw_image, self.input_size)
            # print(depth.max()," =max , min= ",depth.min()," depth map mean = ",depth.mean())
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            depth *= 255.0
            depth = depth.astype(np.uint8)
            
            if self.grayscale:
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            else:
                depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

            # gray_depth = convert_8bit_rgb_to_16bit_grayscale(depth)
            # cv2.imwrite(os.path.join(self.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), gray_depth)
            if self.pred_only:
                cv2.imwrite(os.path.join(self.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)
            else:
                split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
                combined_result = cv2.hconcat([raw_image, split_region, depth])
                
                cv2.imwrite(os.path.join(self.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)

    def aicg_depth_create(self, rgb_images_in=None, save_output_path=None, recon_data: ReconstructData = None):
        '''
        Create a depth map from a set of RGB images
        Input:
            rgb_images_in: List of RGB images
            save_output_path: Path to save the depth map (optional)
        return:
            nd.array: Depth maps
        '''
        if rgb_images_in is None:
            raise ValueError("RGB images must be provided")
        if save_output_path is None:
            save_output_path = default_depth_out_path


        self.img_path = rgb_images_in
        self.outdir = save_output_path

        file_url =  depth_pth_url
        save_directory = default_pth_download_path
        save_filename = "AICG_Depth_Model.pth.downloading"  
        save_path = os.path.join(save_directory, save_filename)
        
        # 获取去掉 .downloading 后缀的文件名
        final_save_path = os.path.join(save_directory, "AICG_Depth_Model.pth")

        os.makedirs(save_directory, exist_ok=True)
        
        # 在下载之前，先检查并删除可能已经存在的 .downloading 文件
        if os.path.exists(save_path):
            os.remove(save_path)

        # 先检查最终文件是否已存在，若存在则跳过下载
        if os.path.exists(final_save_path):
            print(f"File {final_save_path} already exists")
            # 调用回调函数
            if self.on_download_complete:
                self.on_download_complete(final_save_path)
            return
        
        # 调用下载函数
        self.download_file(file_url, save_path, callback=self.on_download_complete)
        pass



class OutdoorsceneReconstruction:
    '''
    A Python API class for outdoor scene reconstruction.
    This class provides a systematic process to:
    1. Extract depth data
    2. Generate 3D point
    3. Reconstruct the 3D scene using multiple inputs and save as FBX or OBJ.
    4. Render new perspective images.
    '''
    def __init__(self,config=None):
        '''
          Input:
              config: Currently Configuration file include Camera extrinsics adn intrinsics etc. this file is enabled to expand other parameters in future.
        '''
        self.__config=config
        self.__depth_creator = DepthCreator(config=self.__config)
        self.__point_creator = PointCreator(config=self.__config)


    def call_class_aicg_depth_create(self, rgb_images_in=None, save_output_path=None, recon_data: ReconstructData = None):
        '''
        Create a depth map from a set of RGB images
        Input:
            rgb_images_in: List of RGB images
            save_output_path: Path to save the depth map (optional)
            recon_data: ReconstructData object containing RGB images, camera extrinsics, camera intrinsics and point cloud (optional)
        return:
            ReconstructData: ReconstructData object containing Depth images
        '''
        return self.__depth_creator.aicg_depth_create(rgb_images_in=rgb_images_in, save_output_path=save_output_path,recon_data=recon_data)

    def call_class_aicg_point_create(self, rgb_images_in=None, save_output_path=None, recon_data: ReconstructData = None):
        '''
        Create a 3D point cloud from a set of RGB images
        Input:
            rgb_images_in: List of RGB images
            save_output_path: Path to save the point cloud
            recon_data: ReconstructData object containing RGB images,depth images, camera extrinsics, camera intrinsics(optional)
        return:
            ReconstructData: ReconstructData object containing RGB images, camera extrinsics, camera intrinsics, and point cloud
        '''
        return self.__point_creator.aicg_point_create(rgb_images_in=rgb_images_in, save_output_path=save_output_path,recon_data=recon_data)

    def aicg_outdoor_render_images(self,reconstruct_model_path=None,point_path_in =None ,save_output_path=None,camera_extrinsics=None,camera_intrinsics=None,recon_data: ReconstructData = None):
        '''
        Rendering new perspective images
        Input:
            reconstruction_model_path: Path to the reconstructed model
            camera_extrinsics: Camera extrinsics
            camera_intrinsics: Camera intrinsics
            save_output_path: Path to save the rendered images
            recon_data: ReconstructData object camera extrinsics, camera intrinsics, and reconstruct_model, expanding in future.(optional)
        return:
            render images (nd.array): Rendered images
        '''
        if save_output_path is None:
            save_output_path = default_outdoor_render_out_path
        if reconstruct_model_path is None:
            reconstruct_model_path = default_outdoor_model_out_path
        if point_path_in is None:
            point_path_in = self.__point_creator.outdir if self.__point_creator.outdir is not None else default_point_out_path
        OutdoorNewViewRender.run(point_path_in,reconstruct_model_path,save_output_path)
        pass

    def aicg_outdoor_mesh_reconstruct(self, point_path_in=None, depth_images_in=None, save_output_path=None, iteration=30000,save_format='.ply', recon_data: ReconstructData = None):
        '''
        Generate a mesh
        Input:
            point_path_in: Path to the point cloud
            depth_images_in: List of depth images
            save_output_path: Path to save the mesh (optional)
            iteration: Number of iterations for the mesh generation
            save_format: Mesh file format (default: '.ply'), The premise is that the value of save_output path is specified.
            recon_data: ReconstructData object containing RGB images,depth images, camera extrinsics, camera intrinsics, and point cloud (optional)
        return:
            mesh (trimesh.Trimesh): Mesh object
        '''
        if point_path_in is None:
            point_path_in = self.__point_creator.outdir if self.__point_creator.outdir is not None else default_point_out_path
        if save_output_path is None:
            save_output_path = default_outdoor_model_out_path
             
        OutdoorReconstruction.run(point_path_in,save_output_path,iteration,save_format)
        pass


class IndoorsceneReconstruction:
    '''
    A Python API class for indoor scene reconstruction.
    This class provides a systematic process to:
    1. Extract depth data
    2. Generate 3D point
    3. Reconstruct the 3D scene using multiple inputs and save as FBX or OBJ.
    4. Render new perspective images.
    '''

    def __init__(self,config=None):
        '''
          Input:
              config: Currently Configuration file include Camera extrinsics adn intrinsics etc. this file is enabled to expand other parameters in future.
        '''
        self.__config=config
        self.__depth_creator = DepthCreator(config=self.__config)
        self.__point_creator = PointCreator(config=self.__config)    

    def call_class_aicg_depth_create(self, rgb_images_in=None, save_output_path=None, recon_data: ReconstructData = None):
        '''
        Create a depth map from a set of RGB images
        Input:
            rgb_images_in: List of RGB images
            save_output_path: Path to save the depth map (optional)
            recon_data: ReconstructData object containing RGB images, camera extrinsics, camera intrinsics and point cloud (optional)
        return:
            ReconstructData: ReconstructData object containing Depth images
        '''
        return self.__depth_creator.aicg_depth_create(rgb_images_in=rgb_images_in, save_output_path=save_output_path,recon_data=recon_data)

    def call_class_aicg_point_create(self, rgb_images_in=None, save_output_path=None, recon_data: ReconstructData = None):
        '''
        Create a 3D point cloud from a set of RGB images
        Input:
            rgb_images_in: List of RGB images
            save_output_path: Path to save the point cloud
            recon_data: ReconstructData object containing RGB images,depth images, camera extrinsics, camera intrinsics(optional)
        return:
            ReconstructData: ReconstructData object containing RGB images, camera extrinsics, camera intrinsics, and point cloud
        '''
        return self.__point_creator.aicg_point_create(rgb_images_in=rgb_images_in, save_output_path=save_output_path,recon_data=recon_data)

    def aicg_indoor_render_images(self,reconstruct_model_path=None,camera_extrinsics=None,camera_intrinsics=None,save_output_path=None,recon_data: ReconstructData = None):
        '''
        Rendering new perspective images
        Input:
            reconstruction_model_path: Path to the reconstructed model
            camera_extrinsics: Camera extrinsics
            camera_intrinsics: Camera intrinsics
            save_output_path: Path to save the rendered images
            recon_data: ReconstructData object camera extrinsics, camera intrinsics, and reconstruct_model, expanding in future.(optional)
        return:
            render images (nd.array): Rendered images
        '''
        pass

    def aicg_indoor_mesh_reconstruct(self, image_path_in=None, depth_images_in=None, save_output_path=None, iteration=30000,save_format='.ply', recon_data: ReconstructData = None):
        '''
        Generate a mesh
        Input:
            image_path_in: List of images
            depth_images_in: List of depth images
            save_output_path: Path to save the mesh (optional)
            iteration: Number of iterations for the mesh generation
            save_format: Mesh file format (default: '.ply'), The premise is that the value of save_output path is specified.
            recon_data: ReconstructData object containing RGB images,depth images, camera extrinsics, camera intrinsics, and point cloud (optional)
        return:
            mesh (trimesh.Trimesh): Mesh object
        '''
        if save_output_path is None:
            save_output_path = default_indoor_model_out_path+"/un_rgb_indoor_model.ply"

        if image_path_in is None:
            print("Please input image path")
            return   
        # load color    
        # color_paths = [image_path_in + "/" + f for f in os.listdir(image_path_in) if f.endswith('.png') or f.endswith('.jpg')]
        color_paths = sorted(glob.glob(os.path.join(image_path_in, "*.png")), key=extract_number)
        # print("color_paths: ",color_paths)
        # load depth
        depth_paths = []
        if depth_images_in is None:
            if os.path.exists(default_depth_out_path):
                 shutil.rmtree(default_depth_out_path)
            self.__depth_creator.aicg_depth_create(rgb_images_in=image_path_in, save_output_path=default_depth_out_path,recon_data=recon_data)
            depth_images_in = default_depth_out_path

        
        # depth_paths = [depth_images_in + "/" + f for f in os.listdir(depth_images_in) if f.endswith('.png')]
        depth_paths = sorted(glob.glob(os.path.join(depth_images_in, "*.png")), key=extract_number)
       
        
        camera_intrinsic =""
        depth_scale = 1000.0
        poses = []
        sence_type = 1
        if recon_data is None:
            if os.path.exists(default_point_out_path):
                 shutil.rmtree(default_point_out_path)
            self.__point_creator.aicg_point_create(rgb_images_in=image_path_in, save_output_path=default_point_out_path,recon_data=recon_data)
            cam_intrinsics = self.__point_creator.colmapConverter.getCameraIntrinsics(default_point_out_path)
            cam_extrinsics = self.__point_creator.colmapConverter.getCameraExtrinsics(default_point_out_path)
            camera_intrinsic=cam_intrinsics
            poses=cam_extrinsics
        else:
            depth_scale = recon_data.get_depth_scale()
            if recon_data.get_camera_intrinsics() is None:
                if os.path.exists(default_point_out_path):
                    shutil.rmtree(default_point_out_path)
                self.__point_creator.aicg_point_create(rgb_images_in=image_path_in, save_output_path=default_point_out_path,recon_data=recon_data)
                cam_intrinsics = self.__point_creator.colmapConverter.getCameraIntrinsics(default_point_out_path)
                cam_extrinsics = self.__point_creator.colmapConverter.getCameraExtrinsics(default_point_out_path)
                camera_intrinsic=cam_intrinsics
                poses=cam_extrinsics
            else:
                camera_intrinsic=recon_data.get_camera_intrinsics()
            if recon_data.get_camera_extrinsics() is not None:
                poses=recon_data.get_camera_extrinsics()
            pass
        if os.path.exists(default_deal_style_out_path):
            shutil.rmtree(default_deal_style_out_path)
        NeuralRGBD.data_style_deal(color_paths=color_paths,depth_paths=depth_paths,poses=poses,camera_intrinsic=camera_intrinsic,output_path=default_deal_style_out_path,depth_scale=depth_scale,sence_type=sence_type)
        ObjectSReconstruction.entrypoint(path_in=default_deal_style_out_path, save_output_path=default_model_out_path,iteration=iteration,sence_type=sence_type)
        print("save_output_path",save_output_path)
        ExtractMesh.entrypoint(default_model_out_path+"/config.yml",save_output_path,sence_type)
        ExtractTextureMesh.entrypoint(default_model_out_path+"/config.yml",save_output_path,default_indoor_model_out_path)
        pass


class ObjectReconstruction:
    '''
    A Python API class for object reconstruction.
    This class provides a systematic process to:
    1. Extract depth data
    2. Generate 3D point
    3. Reconstruct the 3D scene using multiple inputs and save as FBX or OBJ.
    4. Render new perspective images.
    '''

    def __init__(self,config=None):
        '''
          Input:
              config: Currently Configuration file include Camera extrinsics adn intrinsics etc. this file is enabled to expand other parameters in future.
        '''
        self.__config=config
        self.__depth_creator = DepthCreator(config=self.__config)
        self.__point_creator = PointCreator(config=self.__config)

    def call_class_aicg_depth_create(self, rgb_images_in=None, save_output_path=None, recon_data: ReconstructData = None):
        '''
        Create a depth map from a set of RGB images
        Input:
            rgb_images_in: List of RGB images
            save_output_path: Path to save the depth map (optional)
            recon_data: ReconstructData object containing RGB images, camera extrinsics, camera intrinsics and point cloud (optional)
        return:
            ReconstructData: ReconstructData object containing Depth images
        '''
        return self.__depth_creator.aicg_depth_create(rgb_images_in=rgb_images_in, save_output_path=save_output_path,recon_data=recon_data)

    def call_class_aicg_point_create(self, rgb_images_in=None, save_output_path=None, recon_data: ReconstructData = None):
        '''
        Create a 3D point cloud from a set of RGB images
        Input:
            rgb_images_in: List of RGB images
            save_output_path: Path to save the point cloud
            recon_data: ReconstructData object containing RGB images,depth images, camera extrinsics, camera intrinsics(optional)
        return:
            ReconstructData: ReconstructData object containing RGB images, camera extrinsics, camera intrinsics, and point cloud
        '''
        return self.__point_creator.aicg_point_create(rgb_images_in=rgb_images_in, save_output_path=save_output_path,recon_data=recon_data)

    def aicg_object_render_images(self,reconstruct_model_path=None,camera_extrinsics=None,camera_intrinsics=None,save_output_path=None,recon_data: ReconstructData = None):
        '''
        Rendering new perspective images
        Input:
            reconstruction_model_path: Path to the reconstructed model
            camera_extrinsics: Camera extrinsics
            camera_intrinsics: Camera intrinsics
            save_output_path: Path to save the rendered images
            recon_data: ReconstructData object camera extrinsics, camera intrinsics, and reconstruct_model, expanding in future.(optional)
        return:
            render images (nd.array): Rendered images
        '''
        pass

    def aicg_object_mesh_reconstruct(self, image_path_in=None, depth_images_in=None, save_output_path=None, iteration=30000,save_format='.ply', recon_data: ReconstructData = None):
        '''
        Generate a mesh
        Input:
            point_path_in: Path to the point cloud
            depth_images_in: List of depth images
            save_output_path: Path to save the mesh (optional)
            iteration: Number of iterations for the mesh generation
            save_format: Mesh file format (default: '.ply'), The premise is that the value of save_output path is specified.
            recon_data: ReconstructData object containing RGB images,depth images, camera extrinsics, camera intrinsics, and point cloud (optional)
        return:
            mesh (trimesh.Trimesh): Mesh object
        '''
        if save_output_path is None:
            save_output_path = default_object_model_out_path+"/un_rgb_object_model.ply"

        if image_path_in is None:
            print("Please input image path")
            return   
        # load color    
        color_paths = sorted([image_path_in + "/" + f for f in os.listdir(image_path_in) if f.endswith('.png') or f.endswith('.jpg')])
        # color_paths = sorted(color_paths, key=extract_number)
        # load depth
        depth_paths = []
        if depth_images_in is None:
            if os.path.exists(default_depth_out_path):
                 shutil.rmtree(default_depth_out_path)
            self.__depth_creator.aicg_depth_create(rgb_images_in=image_path_in, save_output_path=default_depth_out_path,recon_data=recon_data)
            depth_images_in = default_depth_out_path

        
        depth_paths = sorted([depth_images_in + "/" + f for f in os.listdir(depth_images_in) if f.endswith('.png')])
        # depth_paths = sorted(depth_paths, key=extract_number)
        
        camera_intrinsic =""
        depth_scale = 21.845
        poses = []
        sence_type=0
        if recon_data is None:
            if os.path.exists(default_point_out_path):
                 shutil.rmtree(default_point_out_path)
            self.__point_creator.aicg_point_create(rgb_images_in=image_path_in, save_output_path=default_point_out_path,recon_data=recon_data)
            cam_intrinsics = self.__point_creator.colmapConverter.getCameraIntrinsics(default_point_out_path)
            cam_extrinsics = self.__point_creator.colmapConverter.getCameraExtrinsics(default_point_out_path)
            camera_intrinsic=cam_intrinsics
            poses=cam_extrinsics
        else:
            depth_scale = recon_data.get_depth_scale()
            if recon_data.get_camera_intrinsics() is None:
                if os.path.exists(default_point_out_path):
                    shutil.rmtree(default_point_out_path)
                self.__point_creator.aicg_point_create(rgb_images_in=image_path_in, save_output_path=default_point_out_path,recon_data=recon_data)
                cam_intrinsics = self.__point_creator.colmapConverter.getCameraIntrinsics(default_point_out_path)
                cam_extrinsics = self.__point_creator.colmapConverter.getCameraExtrinsics(default_point_out_path)
                camera_intrinsic=cam_intrinsics
                poses=cam_extrinsics
            else:
                camera_intrinsic=recon_data.get_camera_intrinsics()
            if recon_data.get_camera_extrinsics() is not None:
                poses=recon_data.get_camera_extrinsics()
            pass
        # default_deal_style_out_path = "./input/object/dtu-scan65"
        if os.path.exists(default_deal_style_out_path):
            shutil.rmtree(default_deal_style_out_path)
        NeuralRGBD.data_style_deal(color_paths=color_paths,depth_paths=depth_paths,poses=poses,camera_intrinsic=camera_intrinsic,output_path=default_deal_style_out_path,depth_scale=depth_scale,sence_type=sence_type)
        ObjectSReconstruction.entrypoint(path_in=default_deal_style_out_path, save_output_path=default_model_out_path,iteration=iteration,sence_type=sence_type)
        print("save_output_path",save_output_path)
        ExtractMesh.entrypoint(default_model_out_path+"/config.yml",save_output_path,sence_type)
        ExtractTextureMesh.entrypoint(default_model_out_path+"/config.yml",save_output_path,default_object_model_out_path)
        pass