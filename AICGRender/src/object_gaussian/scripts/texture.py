"""
Script to texture an existing mesh file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch
import torchvision

import sys
import tyro
from rich.console import Console
from typing_extensions import Literal

from ..nerfstudio.exporter import texture_utils
from ..nerfstudio.exporter.exporter_utils import get_mesh_from_filename
from ..nerfstudio.utils.eval_utils import eval_setup
import numpy as np  
import os  

CONSOLE = Console(width=120)


@dataclass
class TextureMesh:
    """
    Export a textured mesh with color computed from the NeRF.
    """

    load_config: Path
    """Path to the config YAML file."""
    output_dir: Path
    """Path to the output directory."""
    input_mesh_filename: Path
    """Mesh filename to texture."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV square."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""
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
        """Export textured mesh"""
        # pylint: disable=too-many-statements

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        # load the Mesh
        mesh = get_mesh_from_filename(str(self.input_mesh_filename), target_num_faces=self.target_num_faces)

        # load the Pipeline
        _, pipeline, _ = eval_setup(self.load_config, test_mode="inference")

        # texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        texture_utils.export_textured_mesh(
            mesh,
            pipeline,
            px_per_uv_triangle=self.px_per_uv_triangle,
            output_dir=self.output_dir,
            unwrap_method=self.unwrap_method,
            num_pixels_per_side=self.num_pixels_per_side,
        )


def entrypoint(mode_config,unrgb_model,save_path):
    # python scripts/texture.py  outputs/neus-facto-dtu65/neus-facto/XXX/config.yml --input-mesh-filename ./meshes/test.ply --output-dir ./textures --target_num_faces 50000
    sys.argv = [
        " ",
        "--load-config", mode_config,
        "--input-mesh-filename", unrgb_model,
        "--output-dir", save_path
        "--target-num-faces", str(50000)
    ]
    print(mode_config+" unrgb_model ",unrgb_model, "  save_path ",save_path)
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[TextureMesh]).main()
