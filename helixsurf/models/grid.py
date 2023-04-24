# Copyright (c) Zhihao Liang. All rights reserved.
from typing import Callable, Optional

import torch
import torch.nn as nn
import open3d as o3d

import helixsurf.libvolume as _C


class OccupancyGrid(nn.Module):
    def __init__(
        self,
        bound: float = 1,
        resolution: int = 64,
        density_thresh: float = 0.01,
        decay: float = 0.9,
    ) -> None:
        """dynamic occupancy grid for sampling

        Args:
            bound (float, optional): bound of aabb. Defaults to 1.
            resolution (int, optional): resolution of aabb. Defaults to 64.
            density_thresh (float, optional): density threshold for judging occupation. Defaults to 0.01.
            decay (float, optional): ema decay. Defaults to 0.9.
        """
        super().__init__()

        # init grid for acceleration
        self.density_grid: torch.Tensor
        density_grid = torch.zeros([resolution] * 3)
        self.register_buffer("density_grid", density_grid)

        self.valid_grid: torch.Tensor
        valid_grid = torch.zeros([resolution] * 3, dtype=torch.bool)
        self.register_buffer("valid_grid", valid_grid)

        self.accel_grid: torch.Tensor
        accel_grid = torch.zeros([resolution] * 3, dtype=torch.int8)
        self.register_buffer("accel_grid", accel_grid)

        self.bound = bound
        self.grid_centers: torch.Tensor
        grid_centers = self.init_grid_centers()
        self.register_buffer("grid_centers", grid_centers)

        self.grid_vertices: torch.Tensor
        grid_vertices = self.init_grid_vertices()
        self.register_buffer("grid_vertices", grid_vertices)

        # update parameters
        self.density_thresh = density_thresh
        self.decay = decay

        # get the seed points for updating
        self.grid_centers = self.init_grid_centers()
        self.grid_vertices = self.init_grid_vertices()

    def init_grid_centers(self) -> torch.Tensor:
        """init the grid centers once for all occupancy update

        Returns:
            torch.Tensor: the centers of grid in world coordinates
        """
        resolution = self.density_grid.shape[0]

        half_grid_size = self.bound / resolution
        device = self.density_grid.device

        X = torch.linspace(
            -self.bound + half_grid_size, self.bound - half_grid_size, resolution
        ).to(device)
        Y = torch.linspace(
            -self.bound + half_grid_size, self.bound - half_grid_size, resolution
        ).to(device)
        Z = torch.linspace(
            -self.bound + half_grid_size, self.bound - half_grid_size, resolution
        ).to(device)
        X, Y, Z = torch.meshgrid(X, Y, Z, indexing="ij")
        return torch.stack((X, Y, Z), dim=-1).view(-1, 3)  # [N, 3]

    def init_grid_vertices(self) -> torch.Tensor:
        """init the grid vertices once for all occupancy update

        Returns:
            torch.Tensor: the centers of grid in world coordinates
        """
        num_vertices = self.density_grid.shape[0] + 1

        device = self.density_grid.device

        X = torch.linspace(-self.bound, self.bound, num_vertices).to(device)
        Y = torch.linspace(-self.bound, self.bound, num_vertices).to(device)
        Z = torch.linspace(-self.bound, self.bound, num_vertices).to(device)
        X, Y, Z = torch.meshgrid(X, Y, Z, indexing="ij")
        return torch.stack((X, Y, Z), dim=-1).view(-1, 3)  # [N, 3]

    def update_occupancy_grid(
        self,
        density_fn: Callable,
        density_thresh: float,
        decay: float = 0.9,
    ) -> None:
        """update the occupancy grid for acceleration

        Args:
            density_fn (Callable): density query function
            density_thresh (float): the threshold for judging occupancy
            decay (float, optional): decay ratio. Defaults to 0.9.
        """
        # call before each epoch to update extra states.
        resolution = self.density_grid.shape[0]
        # update density grid
        centers_shape = (resolution, resolution, resolution)
        vertices_shape = (resolution + 1, resolution + 1, resolution + 1)
        densities_centers = density_fn(self.grid_centers).reshape(centers_shape)  # [128, 128, 128]
        densities_vertices = density_fn(self.grid_vertices).reshape(
            vertices_shape
        )  # [129, 129, 129]

        # voting the density of the grid via the center and 8 vertices
        densities_grid = torch.stack(
            [
                densities_centers,
                densities_vertices[0 : resolution + 0, 0 : resolution + 0, 0 : resolution + 0],
                densities_vertices[0 : resolution + 0, 0 : resolution + 0, 1 : resolution + 1],
                densities_vertices[0 : resolution + 0, 1 : resolution + 1, 0 : resolution + 0],
                densities_vertices[0 : resolution + 0, 1 : resolution + 1, 1 : resolution + 1],
                densities_vertices[1 : resolution + 1, 0 : resolution + 0, 0 : resolution + 0],
                densities_vertices[1 : resolution + 1, 0 : resolution + 0, 1 : resolution + 1],
                densities_vertices[1 : resolution + 1, 1 : resolution + 1, 0 : resolution + 0],
                densities_vertices[1 : resolution + 1, 1 : resolution + 1, 1 : resolution + 1],
            ],
            dim=-1,
        ).max(-1)[
            0
        ]  # [128, 128, 128]

        densities_grid /= densities_grid.max()

        # update
        alpha = decay
        self.density_grid = torch.maximum(self.density_grid * alpha, densities_grid)

        density_thresh = min(torch.mean(self.density_grid).item(), density_thresh)
        self.valid_grid = self.density_grid >= density_thresh
        self.accel_grid = (self.valid_grid).contiguous().to(torch.int8) - 1  # 0 and -1
        _C.accel_dist_prop(self.accel_grid)

    def export_valid_occupancy_grid(
        self,
        density_thresh: float,
        filename: str = "temp.ply",
        return_lines: bool = False,
    ) -> Optional[o3d.geometry.LineSet]:
        """export occupied grids for visualization

        Args:
            density_thresh (float): threshold of density (get valid grid)
            filename (str, optional): path to save results. Defaults to "temp.ply".
            return_lines (bool, optional): return open3d lines or not. Defaults to False.

        Returns:
            Optional[o3d.geometry.LineSet]: reults
        """
        # call before each epoch to update extra states.
        resolution = self.density_grid.shape[0]
        density_thresh = min(torch.mean(self.density_grid).item(), density_thresh)
        valid_grid = self.density_grid >= density_thresh

        from helixsurf.utils.vis import export_valid_grid

        return export_valid_grid(
            resolution=resolution,
            valid_grid=valid_grid,
            lower_bound=[-self.bound] * 3,
            upper_bound=[self.bound] * 3,
            filename=filename,
            return_lines=return_lines,
        )
