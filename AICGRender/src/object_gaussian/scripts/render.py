#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import mediapy as media
import numpy as np
import torch
import tyro
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from typing_extensions import Literal, assert_never

from ..nerfstudio.cameras.camera_paths import (
    generate_ellipse_path,
    get_path_from_json,
    get_spiral_path,
)
import os
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from ..nerfstudio.cameras.camera_paths import get_path_from_json, get_spiral_path
from ..nerfstudio.cameras.cameras import Cameras
from ..nerfstudio.configs.base_config import Config  # pylint: disable=unused-import
from ..nerfstudio.pipelines.base_pipeline import Pipeline
from ..nerfstudio.utils import install_checks
from ..nerfstudio.utils.eval_utils import eval_setup
from ..nerfstudio.utils.rich_utils import ItersPerSecColumn
from ..nerfstudio.data.datamanagers.base_datamanager import AnnotatedDataParserUnion
from ..nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig

CONSOLE = Console(width=120)


def _render_trajectory_video(
    pipeline: Pipeline,
    cameras: Cameras,
    output_filename: Path,
    rendered_output_names: List[str],
    rendered_resolution_scaling_factor: float = 1.0,
    seconds: float = 5.0,
    output_format: Literal["images", "video"] = "images",
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_names: List of outputs to visualise.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Length of output video.
        output_format: How to save output data.
    """
    CONSOLE.print(f"[bold green]Creating new view image {cameras.size}")
    images = []
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    cameras = cameras.to(pipeline.device)

    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )
    output_image_dir = output_filename.parent / output_filename.stem
    if output_format == "images":
        output_image_dir.mkdir(parents=True, exist_ok=True)
    with progress:
        for camera_idx in progress.track(range(cameras.size), description=""):
            camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx)
            with torch.no_grad():
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            render_image = []
            for rendered_output_name in rendered_output_names:
                if rendered_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                    CONSOLE.print(f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center")
                    sys.exit(1)
                output_image = outputs[rendered_output_name].cpu().numpy()
                render_image.append(output_image)
            render_image = np.concatenate(render_image, axis=1)
            if output_format == "images":
                media.write_image(output_image_dir / f"{camera_idx:05d}.png", render_image)
            else:
                images.append(render_image)

    # if output_format == "video":
    #     fps = len(images) / seconds
    #     # make the folder if it doesn't exist
    #     output_filename.parent.mkdir(parents=True, exist_ok=True)
    #     with CONSOLE.status("[yellow]Saving video", spinner="bouncingBall"):
    #         media.write_video(output_filename, images, fps=fps)
    CONSOLE.rule("[green] :tada: :tada: :tada: Success :tada: :tada: :tada:")
    CONSOLE.print(f"[green]Saved video to {output_filename}", justify="center")

def _interpolate_trajectory(cameras: Cameras, num_views: int = 300):
    """calculate interpolate path"""

    c2ws = np.stack(cameras.camera_to_worlds.cpu().numpy())

    key_rots = Rotation.from_matrix(c2ws[:, :3, :3])
    key_times = list(range(len(c2ws)))
    slerp = Slerp(key_times, key_rots)
    interp = interp1d(key_times, c2ws[:, :3, 3], axis=0)
    render_c2ws = []
    for i in range(num_views):
        time = float(i) / num_views * (len(c2ws) - 1)
        cam_location = interp(time)
        cam_rot = slerp(time).as_matrix()
        c2w = np.eye(4)
        c2w[:3, :3] = cam_rot
        c2w[:3, 3] = cam_location
        render_c2ws.append(c2w)
    render_c2ws = torch.from_numpy(np.stack(render_c2ws, axis=0))

    # use intrinsic of first camera
    camera_path = Cameras(
        fx=cameras[0].fx,
        fy=cameras[0].fy,
        cx=cameras[0].cx,
        cy=cameras[0].cy,
        height=cameras[0].height,
        width=cameras[0].width,
        camera_to_worlds=render_c2ws[:, :3, :4],
        camera_type=cameras[0].camera_type,
    )
    return camera_path

@dataclass
class RenderTrajectory:
    """Load a checkpoint, render a trajectory, and save to a video file."""

    # Path to config YAML file.
    load_config: Path
    meatJson: str
    # Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis
    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    #  Trajectory to render.
    traj: Literal["spiral", "filename"] = "spiral"
    # Scaling factor to apply to the camera image resolution.
    downscale_factor: int = 1
    # Filename of the camera path to render.
    camera_path_filename: Path = Path("camera_path.json")
    # Name of the output file.
    output_path: Path = Path("renders/output.mp4")
    # How long the video should be.
    seconds: float = 5.0
    # How to save output data.
    output_format: Literal["images", "video"] = "images"
    # Specifies number of rays per chunk during eval.
    eval_num_rays_per_chunk: Optional[int] = None

    data: AnnotatedDataParserUnion = SDFStudioDataParserConfig()
    num_views: int = 300
    def main(self) -> None:
        """Main function."""
        _, pipeline, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="test" if self.traj == "spiral" else "inference",
        )

        install_checks.check_ffmpeg_installed()

        seconds = self.seconds
        self.data.setup()._setPathData(self.meatJson)
        camera_path = self.data.setup()._generate_dataparser_outputs().cameras
        seconds = camera_path.size / 24
        # if self.traj == "filename":
        #     with open(self.camera_path_filename, "r", encoding="utf-8") as f:
        #         camera_path = json.load(f)
        #     seconds = camera_path["seconds"]
        #     camera_path = get_path_from_json(camera_path)
        # elif self.traj == "interpolate":
        #     outputs = self.data.setup()._generate_dataparser_outputs()
        #     camera_path = _interpolate_trajectory(cameras=outputs.cameras, num_views=self.num_views)
        #     seconds = camera_path.size / 24
        # elif self.traj == "spiral":
        #     outputs = self.data.setup()._generate_dataparser_outputs()
        #     camera_path = get_spiral_path(camera=outputs.cameras, steps=self.num_views, radius=1.0)
        #     seconds = camera_path.size / 24
        # elif self.traj == "ellipse":
        #     outputs = self.data.setup()._generate_dataparser_outputs()
        #     camera_path = generate_ellipse_path(cameras=outputs.cameras, n_frames=self.num_views, const_speed=False)
        #     seconds = camera_path.size / self.fps
        # else:
        #     assert_never(self.traj)
        # TODO(ethan): use camera information from parsing args
        # if self.traj == "spiral":
        #     camera_start = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0).flatten()
        #     # TODO(ethan): pass in the up direction of the camera
        #     camera_path = get_spiral_path(camera_start, steps=30, radius=0.1)
        # elif self.traj == "filename":
        #     with open(self.camera_path_filename, "r", encoding="utf-8") as f:
        #         camera_path = json.load(f)
        #     seconds = camera_path["seconds"]
        #     camera_path = get_path_from_json(camera_path)
        # else:
        #     assert_never(self.traj)

        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=seconds,
            output_format=self.output_format,
        )


def entrypoint(mode_config,save_path):
    root_directory = os.path.abspath(os.path.join(mode_config, "../../"))
    sys.argv = [
        " ",
        "--load-config", mode_config,
        "--meatJson", root_directory+"/data",
        "--output_path", save_path
    ]
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RenderTrajectory).main()
