from  AICGRender.AICGRenderInterface import OutdoorsceneReconstruction,IndoorsceneReconstruction,ObjectReconstruction,ReconstructData
from pathlib import Path
import numpy as np
import os
def load_sdf_poses(path):
    file = open(path, "r")
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

def load_gs_slam_poses(path):
    poses = []
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        pose_ = np.array(list(map(float, line.split()))).reshape(4, 4)
        pose_ = np.linalg.inv(pose_)
        poses.append(pose_)
    return poses    
def test():
  outdoorsceneReconstruction = OutdoorsceneReconstruction()
  # outdoorsceneReconstruction.call_class_aicg_point_create("/home/guowenwu/workspace/packaging_tutorial/input/garden","output/points")
  # outdoorsceneReconstruction.call_class_aicg_depth_create("/home/guowenwu/workspace/packaging_tutorial/input/rgb","output/depth")
  # outdoorsceneReconstruction.aicg_outdoor_mesh_reconstruct("input/garden",iteration=300)

  # indoorsceneReconstruction = IndoorsceneReconstruction()
  # indoorsceneReconstruction.aicg_indoor_mesh_reconstruct(image_path_in = "input/001/out",iteration=40)
  return
  poses=[]      

#   root="input/office2/"
#   pose_path =root+"traj.txt"      
#   poses = load_gs_slam_poses(pose_path)

  # root = "input/001/out/"
  # pose_path =root+"pose_1"   
  # pose_paths = sorted([pose_path + "/" + f for f in os.listdir(pose_path) if f.endswith('.txt')])
  # for pose_path in pose_paths:
  #     c2w = np.loadtxt(pose_path)
  #     # print(c2w)
  #     poses.append(c2w)   

  root = "input/kitchen/"
  pose_path =root+"poses.txt"  
  poses = load_sdf_poses(pose_path)


  # load intrinsic
  intrinsic_path = root+"cam_0.txt"
  camera_intrinsic = np.loadtxt(intrinsic_path)

  recon_data = ReconstructData()    
  recon_data.set_camera_extrinsics(poses)
  recon_data.set_camera_intrinsics(camera_intrinsic)
  recon_data.set_depth_scale(21.845)
  objectReconstruction = ObjectReconstruction()
  objectReconstruction.aicg_object_mesh_reconstruct(image_path_in = root+"rgb",depth_images_in = root+"depth" ,iteration=500,recon_data=recon_data)
  # objectReconstruction.aicg_object_mesh_reconstruct(image_path_in = "input/001/out/rgb",depth_images_in = "output/depth" ,iteration=1000,recon_data=recon_data)
  # objectReconstruction.aicg_object_mesh_reconstruct(image_path_in = "input/001/out/rgb")
test()
