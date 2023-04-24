# Copyright (c) Gorilla-Lab. All rights reserved.
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type, Union

import torch

import helixsurf.libvolume as _C

from .utils.geometry import convert_to_ndc


@dataclass
class RenderOptions:
    """Rendering options, see comments available:"""

    background_brightness: float = 1.0  # [0, 1], the background color black-white

    step_size: float = 0.5  # Step size, in normalized voxels
    #  (i.e. 1 = 1 voxel width, different from svox where 1 = grid width!)

    sigma_thresh: float = 0.01  # Voxels with sigmas < this are ignored, in [0, 1]
    #  make this higher for fast rendering

    stop_thresh: float = (
        1e-7  # Stops rendering if the remaining light intensity/termination, in [0, 1]
    )
    #  probability is <= this much (forward only)
    #  make this higher for fast rendering

    last_sample_opaque: bool = False  # Make the last sample opaque (for forward-facing)

    near_clip: float = 0.0
    use_spheric_clip: bool = False

    """grid parameters"""
    bound: float = 1.0
    density_thresh: float = 0.01
    grid_res: int = 128
    grid_sample_res: int = 128
    max_grid_res: int = 128
    max_grid_sample_res: int = 128

    def _to_cpp(self) -> Type:
        """Generate object to pass to C++

        Returns:
            RenderOptions: the C++ object of rendering options
        """
        opt = _C.RenderOptions()
        opt.background_brightness = self.background_brightness
        opt.step_size = self.step_size
        opt.sigma_thresh = self.sigma_thresh
        opt.stop_thresh = self.stop_thresh
        opt.near_clip = self.near_clip
        opt.use_spheric_clip = self.use_spheric_clip

        opt.last_sample_opaque = self.last_sample_opaque

        opt.bound = self.bound
        opt.density_thresh = self.density_thresh

        opt.grid_res = self.grid_res
        opt.grid_sample_res = self.grid_sample_res

        return opt

    # even the dataclass realize a reliable __repr__, we would like a json style __repr__
    def __repr__(self) -> str:
        content = json.dumps(self.__dict__, indent=4, ensure_ascii=False)
        return content


@dataclass
class Rays:
    origins: torch.Tensor
    dirs: torch.Tensor

    def _to_cpp(self):
        """Generate object to pass to C++

        Returns:
            RaysSpec: the C++ object of rays
        """
        spec = _C.RaysSpec()
        spec.origins = self.origins
        spec.dirs = self.dirs
        return spec

    def __getitem__(self, key) -> "Rays":
        return Rays(self.origins[key], self.dirs[key])

    def __len__(self) -> int:
        return self.origins.size(0)

    @property
    def is_cuda(self) -> bool:
        return self.origins.is_cuda and self.dirs.is_cuda


@dataclass
class Camera:
    c2w: torch.Tensor  # OpenCV
    fx: float = 1111.11
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    width: int = 800
    height: int = 800

    ndc_coeffs: Union[Tuple[float, float], List[float]] = (-1.0, -1.0)

    @property
    def fx_val(self) -> float:
        return self.fx

    @property
    def fy_val(self) -> Optional[float]:
        return self.fx if self.fy is None else self.fy

    @property
    def cx_val(self) -> Optional[float]:
        return self.width * 0.5 if self.cx is None else self.cx

    @property
    def cy_val(self) -> Optional[float]:
        return self.height * 0.5 if self.cy is None else self.cy

    @property
    def using_ndc(self) -> bool:
        return self.ndc_coeffs[0] > 0.0

    def _to_cpp(self):
        """Generate object to pass to C++

        Returns:
            RaysSpec: the C++ object of camera
        """
        spec = _C.CameraSpec()
        spec.c2w = self.c2w
        spec.fx = self.fx_val
        spec.fy = self.fy_val
        spec.cx = self.cx_val
        spec.cy = self.cy_val
        spec.width = self.width
        spec.height = self.height
        spec.ndc_coeffx = self.ndc_coeffs[0]
        spec.ndc_coeffy = self.ndc_coeffs[1]
        return spec

    @property
    def is_cuda(self) -> bool:
        return self.c2w.is_cuda

    def gen_rays(self, px_center: float = 0.0, norm_dir=True) -> Tuple[Rays, float]:
        """Generate the rays for this camera

        Args:
            px_center (float, optional): the partial of x-axis center. Defaults to 0.5.

        Returns:
            Rays: (origins (H*W, 3), dirs (H*W, 3))
            depth_ratio: (float)
        """
        origins = self.c2w[None, :3, 3].expand(self.height * self.width, -1).contiguous()
        yy, xx = torch.meshgrid(
            torch.arange(self.height, dtype=torch.float64, device=self.c2w.device) + px_center,
            torch.arange(self.width, dtype=torch.float64, device=self.c2w.device) + px_center,
            indexing="ij",
        )
        xx = (xx - self.cx_val) / self.fx_val
        yy = (yy - self.cy_val) / self.fy_val
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV
        del xx, yy, zz
        depth_ratio = torch.norm(dirs, dim=-1, keepdim=True)
        if norm_dir:
            dirs = dirs / depth_ratio

        dirs = dirs.reshape(-1, 3, 1)
        dirs = (self.c2w[None, :3, :3].double() @ dirs)[..., 0]
        dirs = dirs.reshape(-1, 3).float()

        if self.ndc_coeffs[0] > 0.0:
            origins, dirs = convert_to_ndc(origins, dirs, self.ndc_coeffs)
            depth_ratio = torch.norm(dirs, dim=-1, keepdim=True)
            if norm_dir:
                dirs = dirs / depth_ratio
        return Rays(origins, dirs), depth_ratio.view(-1, 1).float()
