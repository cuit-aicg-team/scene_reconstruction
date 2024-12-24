from  AICGRender.AICGRenderInterface import OutdoorsceneReconstruction,IndoorsceneReconstruction,ObjectReconstruction,ReconstructData
from pathlib import Path
import numpy as np
import os

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
    
def test():
  outdoorsceneReconstruction = OutdoorsceneReconstruction()
  # outdoorsceneReconstruction.call_class_aicg_point_create("/home/guowenwu/workspace/packaging_tutorial/input/rgb","output/points")
  # outdoorsceneReconstruction.call_class_aicg_depth_create("/home/guowenwu/workspace/packaging_tutorial/input/rgb","output/depth")
  # outdoorsceneReconstruction.aicg_outdoor_mesh_reconstruct("output/points",iteration=100)

  # indoorsceneReconstruction = IndoorsceneReconstruction()
  # indoorsceneReconstruction.aicg_indoor_mesh_reconstruct(image_path_in = "input/001/out",iteration=40)
  
  pose_path ="input/001/out/pose_1"      
  poses=[]      
  pose_paths = sorted([pose_path + "/" + f for f in os.listdir(pose_path) if f.endswith('.txt')])
  for pose_path in pose_paths:
      c2w = np.loadtxt(pose_path)
      # print(c2w)
      poses.append(c2w)   

  # load intrinsic
  intrinsic_path = "input/001/out/cam_0.txt"
  camera_intrinsic = np.loadtxt(intrinsic_path)

  recon_data = ReconstructData()    
  recon_data.set_camera_extrinsics(poses)
  recon_data.set_camera_intrinsics(camera_intrinsic)
  recon_data.set_depth_scale(255.0)
  objectReconstruction = ObjectReconstruction()
  objectReconstruction.aicg_object_mesh_reconstruct(image_path_in = "input/001/out/rgb",depth_images_in = "input/001/out/depth" ,iteration=1000,recon_data=recon_data)
  # objectReconstruction.aicg_object_mesh_reconstruct(image_path_in = "input/001/out/rgb",depth_images_in = "output/depth" ,iteration=1000,recon_data=recon_data)
  # objectReconstruction.aicg_object_mesh_reconstruct(image_path_in = "input/001/out/rgb")
test()
