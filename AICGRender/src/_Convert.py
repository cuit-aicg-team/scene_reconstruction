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

import os
import logging
import shutil

from .colmap_loader import read_extrinsics_text, read_intrinsics_text,read_extrinsics_binary, read_intrinsics_binary
class ColmapConverter(object):
    def __init__(self, args=None):
        if args is None:
            return
        self.args = args
        self.source_path = args["source_path"]
        self.save_path = args["save_path"]
        self.camera = "OPENCV"
        self.resize = False
        self.skip_matching = False
        self.colmap_command = '"{}"'.format("colmap")
        self.use_gpu = "on"

    def convert(self):
        if not self.skip_matching:
            os.makedirs(self.save_path + "/distorted/sparse", exist_ok=True)

            ## Feature extraction
            feat_extracton_cmd = self.colmap_command + " feature_extractor "\
                "--database_path " + self.save_path + "/distorted/database.db \
                --image_path " + self.source_path + " \
                --ImageReader.single_camera 1 \
                --ImageReader.camera_model " + self.camera+"\
                --SiftExtraction.use_gpu " + str(self.use_gpu)
            exit_code = os.system(feat_extracton_cmd)
            if exit_code != 0:
                logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
                exit(exit_code)

            ## Feature matching
            feat_matching_cmd = self.colmap_command + " exhaustive_matcher \
                --database_path " + self.save_path + "/distorted/database.db \
                --SiftMatching.use_gpu " + str(self.use_gpu)
            exit_code = os.system(feat_matching_cmd)
            if exit_code != 0:
                logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
                exit(exit_code)

            ### Bundle adjustment
            # The default Mapper tolerance is unnecessarily large,
            # decreasing it speeds up bundle adjustment steps.
            mapper_cmd = (self.colmap_command + " mapper \
                --database_path " + self.save_path + "/distorted/database.db \
                --image_path "  + self.source_path + " \
                --output_path "  + self.save_path + "/distorted/sparse \
                --Mapper.ba_global_function_tolerance=0.000001")
            exit_code = os.system(mapper_cmd)
            if exit_code != 0:
                logging.error(f"Mapper failed with code {exit_code}. Exiting.")
                exit(exit_code)

        ### Image undistortion
        ## We need to undistort our images into ideal pinhole intrinsics.
        img_undist_cmd = (self.colmap_command + " image_undistorter \
            --image_path " + self.source_path + " \
            --input_path " + self.save_path + "/distorted/sparse/0 \
            --output_path " + self.save_path + "\
            --output_type COLMAP")
        exit_code = os.system(img_undist_cmd)
        if exit_code != 0:
            logging.error(f"Mapper failed with code {exit_code}. Exiting.")
            exit(exit_code)

        files = os.listdir(self.save_path + "/sparse")
        os.makedirs(self.save_path + "/sparse/0", exist_ok=True)
        # Copy each file from the source directory to the destination directory
        for file in files:
            if file == '0':
                continue
            source_file = os.path.join(self.save_path, "sparse", file)
            destination_file = os.path.join(self.save_path, "sparse", "0", file)
            shutil.move(source_file, destination_file)

        if(self.resize):
            print("Copying and resizing...")

            # Resize images.
            os.makedirs(self.save_path + "/images_2", exist_ok=True)
            os.makedirs(self.save_path + "/images_4", exist_ok=True)
            os.makedirs(self.save_path + "/images_8", exist_ok=True)
            # Get the list of files in the source directory
            files = os.listdir(self.save_path + "/images")
            # Copy each file from the source directory to the destination directory
            # for file in files:
            #     source_file = os.path.join(self.save_path, "images", file)

            #     destination_file = os.path.join(self.save_path, "images_2", file)
            #     shutil.copy2(source_file, destination_file)
            #     exit_code = os.system(self.magick_command + " mogrify -resize 50% " + destination_file)
            #     if exit_code != 0:
            #         logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            #         exit(exit_code)

            #     destination_file = os.path.join(self.save_path, "images_4", file)
            #     shutil.copy2(source_file, destination_file)
            #     exit_code = os.system(self.magick_command + " mogrify -resize 25% " + destination_file)
            #     if exit_code != 0:
            #         logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            #         exit(exit_code)

            #     destination_file = os.path.join(self.save_path, "images_8", file)
            #     shutil.copy2(source_file, destination_file)
            #     exit_code = os.system(self.magick_command + " mogrify -resize 12.5% " + destination_file)
            #     if exit_code != 0:
            #         logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            #         exit(exit_code)

        print("Done.")
    def getCameraExtrinsics(self,path=None):
        if path is None:
            path = self.save_path
        print("Getting extrinsics...",path)    
        try:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        return cam_extrinsics    
    
    def getCameraIntrinsics(self,path=None):
        if path is None:
            path = self.save_path
        try:
          cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
          cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
          cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
          cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
        return cam_intrinsics    