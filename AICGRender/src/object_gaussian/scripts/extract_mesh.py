#!/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple
import json  
import os  
import torch
import numpy as np  
import tyro
import sys
from rich.console import Console

from ..nerfstudio.model_components.ray_samplers import save_points
from ..nerfstudio.utils.eval_utils import eval_setup
from ..nerfstudio.utils.marching_cubes import (
    get_surface_occupancy,
    get_surface_sliding,
    get_surface_sliding_with_contraction,
)

CONSOLE = Console(width=120)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore


@dataclass
class ExtractMesh:
    """Load a checkpoint, run marching cubes, extract mesh, and save it to a ply file."""

    # Path to config YAML file.
    load_config: Path
    gaussian_config: Path = None
    # Marching cube resolution.
    resolution: int = 1024
    # Name of the output file.
    output_path: Path = Path("meshes")
    # Whether to simplify the mesh.
    simplify_mesh: bool = False
    # extract the mesh using occupancy field (unisurf) or SDF, default sdf
    is_occupancy: bool = False
    """Minimum of the bounding box."""
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
    """Maximum of the bounding box."""
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """marching cube threshold"""
    marching_cube_threshold: float = 0.0
    """create visibility mask"""
    create_visibility_mask: bool = False
    """save visibility grid"""
    save_visibility_grid: bool = False
    """visibility grid resolution"""
    visibility_grid_resolution: int = 512
    """threshold for considering a points is valid when splat to visibility grid"""
    valid_points_thres: float = 0.005
    """sub samples factor of images when creating visibility grid"""
    sub_sample_factor: int = 8
    """torch precision"""
    torch_precision: Literal["highest", "high"] = "high"

    def getGt(self):
        original_path = str(self.load_config)
        new_path = original_path.replace('config.yml','gt.txt')  
        matrix = []  
        if os.path.exists(new_path):  
            with open(new_path, 'r') as file:  
                lines = file.readlines()  
                index = 0
                numbers =[]
                for line in lines:  
                    # 去除每行末尾的换行符，并按空格分割字符串为列表  
                    number_str = line.strip().split()  
                    
                    # 将字符串列表转换为整数列表  
                    number = float(number_str[-1])
                    numbers.append(number)
                    # 确保每行有4个数字  
                    if (index+1)%4==0:
                       matrix.append(numbers)  
                       numbers =[]
                    index+=1   
                   
                # 检查矩阵是否完整  
                if len(matrix) == 4:  
                    # 矩阵已经是4x4的，可以直接使用  
                    print("Matrix:")  
                    for row in matrix:  
                        print(row)  
                else:  
                    print("Error: The file does not contain exactly 16 numbers (4x4 matrix).")
                    return None  
        else:  
            print(f"File {new_path} does not exist.")
            return None

        return np.array(matrix)    

    def main(self) -> None:
        """Main function."""
        torch.set_float32_matmul_precision(self.torch_precision)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        _, pipeline, _ = eval_setup(self.load_config)
        CONSOLE.print("Extract mesh with marching cubes and may take a while")
        # print("new point",self.new_view_path)
        # 获取worldtogt  
        worldtogt = self.getGt()

        if self.create_visibility_mask:
            assert self.resolution % 512 == 0

            coarse_mask = pipeline.get_visibility_mask(
                self.visibility_grid_resolution, self.valid_points_thres, self.sub_sample_factor
            )

            def inv_contract(x):
                mag = torch.linalg.norm(x, ord=pipeline.model.scene_contraction.order, dim=-1)
                mask = mag >= 1
                x_new = x.clone()
                x_new[mask] = (1 / (2 - mag[mask][..., None])) * (x[mask] / mag[mask][..., None])
                return x_new

            if self.save_visibility_grid:
                offset = torch.linspace(-2.0, 2.0, 512)
                x, y, z = torch.meshgrid(offset, offset, offset, indexing="ij")
                offset_cube = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(coarse_mask.device)
                points = offset_cube[coarse_mask.reshape(-1) > 0]
                points = inv_contract(points)
                save_points("mask.ply", points.cpu().numpy())
                torch.save(coarse_mask, "coarse_mask.pt")

            get_surface_sliding_with_contraction(
                sdf=lambda x: (
                    pipeline.model.field.forward_geonetwork(x)[:, 0] - self.marching_cube_threshold
                ).contiguous(),
                resolution=self.resolution,
                bounding_box_min=self.bounding_box_min,
                bounding_box_max=self.bounding_box_max,
                coarse_mask=coarse_mask,
                output_path=self.output_path,
                simplify_mesh=self.simplify_mesh,
                inv_contraction=inv_contract,
            )
            return

        if self.is_occupancy:
            # for unisurf
            get_surface_occupancy(
                occupancy_fn=lambda x: torch.sigmoid(
                    10 * pipeline.model.field.forward_geonetwork(x)[:, 0].contiguous()
                ),
                resolution=self.resolution,
                bounding_box_min=self.bounding_box_min,
                bounding_box_max=self.bounding_box_max,
                level=0.5,
                device=pipeline.model.device,
                output_path=self.output_path,
            )
        else:
            assert self.resolution % 512 == 0
            
            # for sdf we can multi-scale extraction.
            get_surface_sliding(
                sdf=lambda x: pipeline.model.field.forward_geonetwork(x)[:, 0].contiguous(),
                resolution=self.resolution,
                bounding_box_min=self.bounding_box_min,
                bounding_box_max=self.bounding_box_max,
                coarse_mask=pipeline.model.scene_box.coarse_binary_gird,
                output_path=self.output_path,
                simplify_mesh=self.simplify_mesh,
                gt=worldtogt,
                gaussian_config=self.gaussian_config
            )


def entrypoint(default_config_path,save_output_path):
    sys.argv = [
        "ns-extract-mesh", 
        "--load-config", default_config_path,
        "--output-path", save_output_path,
        "--gaussian_config" ,"AICGRender/src/object_gaussian/nerfstudio/configs/game1.yaml",
    ]
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[ExtractMesh]).main()