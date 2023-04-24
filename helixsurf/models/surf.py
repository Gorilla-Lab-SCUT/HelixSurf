# Copyright (c) Gorilla-Lab. All rights reserved.
import os
from typing import Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F

import helixsurf.libvolume as _C

from ..primitives import Camera, Rays, RenderOptions
from .embedder import PositionalEncoding, SphericalEncoding
from .grid import OccupancyGrid
from .scatter import scatter_cumprod, scatter_sum_2d, scatter_sum_broadcast


class HelixSurf(nn.Module):
    def __init__(
        self,
        bound: float,
        grid_res: int,
        level: int = 10,
        sh_degree: int = 4,
        sdf_n_layers: int = 8,
        sdf_skip_in: Sequence[int] = (4,),
        rgb_n_layers: int = 4,
        dim_geo: int = 64,
        init_val: float = 0.3,
        inside_outside: bool = True,
        **kwargs,
    ) -> None:
        """initialize the model

        Args:
            bound (float): field bound
            grid_res (int): resolution of the acceleration grid
            level (int, optional): level of positional encoding. Defaults to 10.
            sh_degree (int, optional): degree of spherical encoding. Defaults to 4.
            sdf_n_layers (int, optional): the numbers of layers in SDF network. Defaults to 8.
            sdf_skip_in (Sequence[int], optional): the ids of skip layers in SDF network. Defaults to (4,).
            rgb_n_layers (int, optional): the numbers of layers in rgb network. Defaults to 4.
            dim_geo (int, optional): dimension of the geometry feature. Defaults to 64.
            init_val (float, optional): init val of single variance. Defaults to 0.3.
        """
        super().__init__()

        self.sdf_net: SDFNetwork
        self.rgb_net: RGBNetwork
        self.deviation_net = SingleVarianceNetwork(init_val)

        self.bound = bound
        self.occupancy_grid = OccupancyGrid(bound=bound, resolution=grid_res)

        self.sh_degree = sh_degree
        self.sdf_dim_output = dim_geo + 1
        self.rgb_dim_input = 3 + 3 + sh_degree**2 + dim_geo

        # TODO: this if for pure python implementation(useless, to remove)
        self.n_samples = 256

        self.init_sdf_net(
            dim_hidden=dim_geo,
            n_layers=sdf_n_layers,
            skip_in=sdf_skip_in,
            level=level,
            inside_outside=inside_outside,
            bias=1.0,
        )
        self.init_rgb_net(dim_hidden=dim_geo, n_layers=rgb_n_layers, sh_degree=self.sh_degree)

    def forward(
        self,
        pos_input: torch.Tensor,
        dir_input: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        with torch.enable_grad():
            x, normals = self.sdf_net(pos_input, with_normals=True)  # [num_samples, 1 + geo_dim]
        sdf = x[..., :1]
        geo_feat = x[..., 1:]

        colors = self.rgb_net(pos_input, normals, dir_input, geo_feat)  # [B, output_dim]

        inv_s = self.deviation_net(torch.zeros([1, 3])).clip(1e-6, 1e6)  # Single parameter
        inv_s = inv_s.expand(len(pos_input), 1)

        if colors.shape[-1] == 3:  # for rgb output, use sigmoid function
            colors = torch.sigmoid(colors)

        sdf = sdf.contiguous()
        colors = colors.contiguous()

        return colors, sdf, normals, inv_s

    @torch.no_grad()
    def sampling(
        self,
        rays: Rays,
        opt: RenderOptions,
        n_importance: int = 8,
        up_sample_steps: int = 1,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        """sampling points for each ray to forward

        Args:
            rays (Rays): rays for sampling
            opt (RenderOptions): rendering options
            n_importance (int, optional): number of importance points for up sampling. Defaults to 8.
            up_sample_steps (int, optional): times of upsampling. Defaults to 1.

        Returns:
            offsets (torch.Tensor, [n_rays + 1]): offsets for scattering
            pos_idrs (torch.Tensor, [n_sample, 6]): position and directions for each sample
            deltas (torch.Tensor, [n_sample]): deltas for each sample
            steps (torch.Tensor, [n_sample]): depth for each sample on the corresponding ray
        """
        spec_rays = rays._to_cpp()
        spec_opt = opt._to_cpp()

        offsets, pos_dirs, deltas, steps = _C.marching_rays(
            self.occupancy_grid.valid_grid, self.occupancy_grid.accel_grid, spec_rays, spec_opt
        )

        for i in range(up_sample_steps):
            assert n_importance > 0
            pos_input = pos_dirs[:, :3]
            dir_input = pos_dirs[:, 3:]
            colors, sdf, normals, inv_s = self(pos_input, dir_input)

            mid_sdf, diff_sdf, dist, cos_val, diff_offsets = _C.prev_next_diff(
                sdf.contiguous(), steps.contiguous(), offsets
            )
            prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
            next_esti_sdf = mid_sdf + cos_val * dist * 0.5

            inv_variance = 64 * 2**i
            prev_cdf = torch.sigmoid(prev_esti_sdf * inv_variance)
            next_cdf = torch.sigmoid(next_esti_sdf * inv_variance)

            alphas = ((prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)).clip(0.0, 1.0)

            trans = scatter_cumprod(1.0 - alphas + 1e-7, diff_offsets, True)
            weights = trans * alphas
            # normalize weights
            weights = weights / (scatter_sum_broadcast(weights, diff_offsets) + 1e-10)

            offsets, steps, pos_dirs, deltas, importance_steps = _C.up_sample(
                self.valid_grid,
                steps.reshape(-1),
                weights.reshape(-1),
                offsets,
                diff_offsets,
                n_importance,
                spec_rays,
                spec_opt,
            )

        return offsets, pos_dirs, deltas, steps

    def train_render(
        self,
        rays: Rays,
        opt: RenderOptions,
        cos_anneal_ratio: float = 0.0,
        ek_bound: float = 2.0,
        norm_weight_for_depth: bool = True,
        render_normal: bool = False,
        n_importance: int = 8,
        up_sample_steps: int = 1,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        """CUDA implementation of training volume rendering.

        Args:
            rays (Rays): input rays
            opt (RenderOptions): rendering options
            cos_anneal_ratio (float, optional):
                cos anneal ratio value. Defaults to 0.0.
            ek_bound (float, optional):
                the bound to sample points for eikonal loss. Default to 2.0
            norm_weight_for_depth (bool, optional):
                normalize weight for each ray or not. Defaults to True.
            render_normal (bool, optional):
                calculate normal via volume rendering or surface point backward. Defaults to False.
            n_importance (int, optional): number of importance points for up sampling. Defaults to 8.
            up_sample_steps (int, optional): times of upsampling. Defaults to 1.

        Returns:
            rgb_out (torch.Tensor, [n_rays, 3]): predicted rgb of rays
            depth_out (torch.Tensor, [n_rays]): predicted depth of rays
            normal_out (torch.Tensor, [n_rays, 3]): predicted normal of rays
            normal_eik (torch.Tensor, []): sample points for calculating eikonal loss
            inv_s (torch.Tensor, []): predicted inv single variance
            surface_points (torch.Tensor, [n_rays, 3]): predicted surface points of rays
        """

        offsets, pos_dirs, deltas, steps = self.sampling(rays, opt, n_importance, up_sample_steps)

        pos_input = pos_dirs[:, :3]
        dir_input = pos_dirs[:, 3:]
        colors, sdf, normals, inv_s = self(pos_input, dir_input)

        # from neus
        true_cos = (dir_input * normals).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio)
            + F.relu(-true_cos) * cos_anneal_ratio
        )

        # Estimate signed distances at section points
        estimated_prev_sdf = sdf - iter_cos * deltas.view(-1, 1) * 0.5
        estimated_next_sdf = sdf + iter_cos * deltas.view(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        alphas = ((prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)).clip(0.0, 1.0)

        trans = scatter_cumprod(1.0 - alphas + 1e-7, offsets, True)
        weights = trans * alphas

        weighted_colors = colors.reshape(-1, 3) * weights.reshape(-1, 1)
        rgb_out = scatter_sum_2d(weighted_colors, offsets)

        weights_ = (
            weights / (scatter_sum_broadcast(weights, offsets) + 1e-10)
            if norm_weight_for_depth
            else weights
        )
        depth_out = scatter_sum_2d(steps.reshape(-1, 1) * weights_.reshape(-1, 1), offsets)

        surface_points = rays.origins + rays.dirs * depth_out.view(-1, 1)
        peak_weights_mask = _C.scatter_max(weights, offsets)
        peak_weights_points = pos_input[peak_weights_mask]

        if render_normal:
            normal_out = scatter_sum_2d(normals * weights_.reshape(-1, 1), offsets)
        else:
            _, normal_out = self.sdf_net(surface_points.detach())

        """ For Eikonal loss """
        eikonal_points = torch.empty_like(peak_weights_points).uniform_(-ek_bound, ek_bound)
        eikonal_points = torch.cat([peak_weights_points.detach(), eikonal_points], dim=-2)
        _, normal_eik = self.sdf_net(eikonal_points)
        normal_eik = torch.cat([normal_eik, normal_out], dim=-2)
        normal_eik = torch.norm(normal_eik, p=2, dim=-1)

        return (
            rgb_out,
            depth_out,
            normal_out,
            normal_eik,
            (1.0 / inv_s).mean(),
            surface_points,
        )

    def eval_render(
        self,
        opt: RenderOptions,
        camera: Camera,
        batch_size: int = 500,
        cos_anneal_ratio: float = 0.0,
        norm_weight_for_depth: bool = True,
        render_normal: bool = False,
        n_importance: int = 8,
        up_sample_steps: int = 1,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        """CUDA implementation of volume rendering (entire image version).

        Args:
            opt (RenderOptions): rendering options.
            camera (Camera): input camera.
            batch_size (int, optional): batch size for prediction. Defaults to 500.
            cos_anneal_ratio (float, optional): cos anneal ratio value. Defaults to 0.0.
            norm_weight_for_depth (bool, optional):
                normalize weight for each ray or not. Defaults to True.
            render_normal (bool, optional):
                calculate normal via volume rendering or surface point backward. Defaults to False.
            n_importance (int, optional): number of importance points for up sampling. Defaults to 8.
            up_sample_steps (int, optional): times of upsampling. Defaults to 1.

        Returns:
            all_rgb_out (torch.Tensor, [H, W, 3]): predicted rgb image of the given camera.
            all_depth_out (torch.Tensor, [H, W]): predicted depth map of the given camera.
            all_normal_out (torch.Tensor, [H, W, 3]): predicted normal map of the given camera.
            all_surf_pts (torch.Tensor, [H, W, 3]): predicted surface points of the given camera.
        """

        rays, depth_ratio = camera.gen_rays(px_center=0.0)  # for dtu
        all_rgb_out = []
        all_depth_out = []
        all_normal_out = []
        all_surf_pts = []

        self.occupancy_grid.accel_grid = (self.occupancy_grid.valid_grid).contiguous().to(
            torch.int8
        ) - 1  # 0 and -1
        _C.accel_dist_prop(self.occupancy_grid.accel_grid)

        for batch_start in range(0, camera.height * camera.width, batch_size):
            batch_rays = rays[batch_start : batch_start + batch_size]
            offsets, pos_dirs, deltas, steps = self.sampling(
                batch_rays, opt, n_importance, up_sample_steps
            )

            pos_input = pos_dirs[:, :3]
            dir_input = pos_dirs[:, 3:]
            colors, sdf, normals, inv_s = self(pos_input, dir_input)

            true_cos = (dir_input * normals).sum(-1, keepdim=True)

            iter_cos = -(
                F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio)
                + F.relu(-true_cos) * cos_anneal_ratio
            )

            # Estimate signed distances at section points
            estimated_prev_sdf = sdf - iter_cos * deltas.view(-1, 1) * 0.5
            estimated_next_sdf = sdf + iter_cos * deltas.view(-1, 1) * 0.5

            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

            alphas = ((prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)).clip(0.0, 1.0)

            trans = scatter_cumprod(1.0 - alphas + 1e-7, offsets, True)
            weights = trans * alphas

            weighted_colors = colors.view(-1, 3) * weights.view(-1, 1)
            rgb_out_part = scatter_sum_2d(weighted_colors, offsets)

            weights_ = (
                weights / (scatter_sum_broadcast(weights, offsets) + 1e-10)
                if norm_weight_for_depth
                else weights
            )
            depth_out = scatter_sum_2d(steps.reshape(-1, 1) * weights_.reshape(-1, 1), offsets)

            batch_surface_points = batch_rays.origins + batch_rays.dirs * depth_out.view(-1, 1)
            if render_normal:
                normal_out = scatter_sum_2d(normals * weights_.reshape(-1, 1), offsets)
            else:
                with torch.enable_grad():
                    _, normal_out = self.sdf_net(batch_surface_points.detach())

            all_rgb_out.append(rgb_out_part)
            all_depth_out.append(depth_out)
            all_normal_out.append(normal_out)
            all_surf_pts.append(batch_surface_points)

        all_rgb_out = torch.cat(all_rgb_out, dim=0)
        all_depth_out = (
            torch.cat(all_depth_out, dim=0) / depth_ratio
        )  # 　Rescale the depth, aligned to gt
        all_normal_out = torch.cat(all_normal_out, dim=0)
        all_surf_pts = torch.cat(all_surf_pts, dim=0)

        return (
            all_rgb_out.view(camera.height, camera.width, 3),
            all_depth_out.view(camera.height, camera.width),
            all_normal_out.view(camera.height, camera.width, 3),
            all_surf_pts.view(camera.height, camera.width, 3),
        )

    @torch.no_grad()
    def density(self, pos_input: torch.Tensor, return_sdf: bool = False) -> torch.Tensor:
        """predict the density-sigma/sdf of the input points

        Args:
            pos_input (torch.Tensor, [n_sample, 3]):
                the position of input points
            return_sdf (bool, optional):
                return the predicted sdf value or not. Defaults to False.

        Returns:
            (torch.Tensor, [n_sample]): predicted density of sdf value
        """
        sdf = self.sdf_net.sdf(pos_input)  # [num_samples]
        if return_sdf:
            return sdf
        inv_s = self.deviation_net(torch.zeros([1, 3])).clip(1e-6, 1e6)  # Single parameter
        inv_s = inv_s.expand(len(sdf), 1)
        exp = torch.exp(-inv_s * sdf).clip(1e-10, 1e10)
        density = inv_s * exp / (1 + exp) ** 2

        if torch.count_nonzero(density.isnan()) > 0:
            raise ValueError

        return density.view(-1)

    def init_sdf_net(self, **kwargs) -> None:
        kwargs.update(dict(dim_output=self.sdf_dim_output))
        self.sdf_net = SDFNetwork(**kwargs)

    def init_rgb_net(self, **kwargs) -> None:
        kwargs.update(dict(dim_input=self.rgb_dim_input))
        self.rgb_net = RGBNetwork(**kwargs)

    def init_grid_centers(self) -> torch.Tensor:
        """init the grid centers once for all occupancy update

        Returns:
            torch.Tensor: the centers of grid in world coordinates
        """
        resolution = self.occupancy_grid.shape[0]

        half_grid_size = self.bound / resolution
        device = self.occupancy_grid.density_grid.device

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
        num_vertices = self.occupancy_grid.shape[0] + 1

        device = self.occupancy_grid.device

        X = torch.linspace(-self.bound, self.bound, num_vertices).to(device)
        Y = torch.linspace(-self.bound, self.bound, num_vertices).to(device)
        Z = torch.linspace(-self.bound, self.bound, num_vertices).to(device)
        X, Y, Z = torch.meshgrid(X, Y, Z, indexing="ij")
        return torch.stack((X, Y, Z), dim=-1).view(-1, 3)  # [N, 3]

    def update_occupancy_grid(
        self,
        decay: float = 0.9,
        opt: Optional[RenderOptions] = None
    ) -> None:
        """update the occupancy grid for acceleration

        Args:
            decay (float, optional): decay ratio. Defaults to 0.9.
            opt (Optional[RenderOptions], optional): rendering options. Defaults to None.
        """
        self.occupancy_grid.update_occupancy_grid(
            self.density, opt.density_thresh, decay
        )

    def export_valid_occupancy_grid(
        self,
        opt: RenderOptions,
        filename: str = "temp.ply",
        return_lines: bool = False,
    ) -> Optional[o3d.geometry.LineSet]:
        """export occupied grids for visualization

        Args:
            opt (RenderOptions): rendering options
            filename (str, optional): path to save result. Defaults to "temp.ply".
            return_lines (bool, optional): return open3d lines or not. Defaults to False.

        Returns:
            Optional[o3d.geometry.LineSet]: reults
        """
        return self.occupancy_grid.export_valid_occupancy_grid(
            opt.density_thresh, filename, return_lines
        )

    @torch.no_grad()
    def export_mesh(
        self,
        filename: str = "vis_mc_uniform/sdf_mc.ply",
        resolution: int = 64,
        batch_size: int = 64**3,
        thresh: float = 0,
    ) -> None:
        """export mesh for visualization

        Args:
            filename (str, optional): path to save result. Defaults to "vis_mc_uniform/sdf_mc.ply".
            resolution (int, optional): marching cube resolution. Defaults to 64.
            batch_size (int, optional): batch size of marching cubes querying. Defaults to 64**3.
            thresh (float, optional): threshold for marching cubes. Defaults to 0.
        """
        centers_shape = (resolution, resolution, resolution)
        half_grid_size = self.bound / resolution
        device = self.occupancy_grid.density_grid.device

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
        mc_grid = torch.stack((X, Y, Z), dim=-1).view(-1, 3)  # [N, 3]

        import prim3d

        save_dir = os.path.dirname(filename)
        os.makedirs(save_dir, exist_ok=True)
        sdf = torch.zeros_like(mc_grid[..., 0])
        for i in range(0, len(mc_grid), batch_size):
            sdf[i : i + batch_size] = self.sdf_net.sdf(mc_grid[i : i + batch_size])[..., 0]
        sdf = sdf.reshape(centers_shape).contiguous()
        vertices, faces = prim3d.marching_cubes(
            sdf, thresh, ([-self.bound] * 3, [self.bound] * 3), verbose=False
        )
        prim3d.save_mesh(vertices, faces, filename=filename)

        torch.cuda.empty_cache()


class SDFNetwork(nn.Module):
    def __init__(
        self,
        dim_output: int,
        dim_hidden: int = 256,
        n_layers: int = 8,
        skip_in: Sequence[int] = (4,),
        level: int = 10,
        geometric_init: bool = True,
        weight_norm: bool = True,
        inside_outside: bool = False,
        bias: float = 1.0,
    ) -> None:
        super().__init__()

        self.frequency_encoder = PositionalEncoding(level)

        input_ch = level * 6 + 3

        dims = [input_ch] + [dim_hidden for _ in range(n_layers)] + [dim_output]

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=-np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        torch.nn.init.constant_(lin.bias, bias)
                elif level > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif level > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
            if l < self.num_layers - 2:
                setattr(self, "act" + str(l), nn.Softplus(beta=100))
                # setattr(self, "act" + str(l), nn.ReLU(inplace=True))

    def forward(self, inputs: torch.Tensor, with_normals: bool = True) -> torch.Tensor:
        if with_normals:
            inputs.requires_grad_(True)

        x = self.frequency_encoder(inputs)
        x_enc = x
        for l in range(0, self.num_layers - 1):
            lin: nn.Linear = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, x_enc], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = getattr(self, "act" + str(l))(x)

        if not with_normals:
            return x
        else:
            sdf = x[:, :1]
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            normals = torch.autograd.grad(
                outputs=sdf,
                inputs=inputs,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            return x, normals

    def sdf(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, with_normals=False)[:, :1]


class RGBNetwork(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_hidden: int = 256,
        n_layers: int = 4,
        sh_degree: int = 4,
        weight_norm: bool = True,
    ) -> None:
        super().__init__()

        self.spherical_encoder = SphericalEncoding(sh_degree)

        dims = [dim_input] + [dim_hidden for _ in range(n_layers)] + [3]
        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
            if l < self.num_layers - 2:
                setattr(self, "act" + str(l), nn.ReLU(inplace=True))

    def forward(
        self,
        points: torch.Tensor,
        normals: torch.Tensor,
        dirs: torch.Tensor,
        geo_feature: torch.Tensor,
    ) -> torch.Tensor:
        # Note that tcnn SH encoding requires inputs to be in [0, 1]
        dirs_encoded = self.spherical_encoder((dirs + 1) / 2)

        rendering_input = torch.cat([points, dirs_encoded, normals, geo_feature], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = getattr(self, "act" + str(l))(x)

        return x


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val: float) -> None:
        super().__init__()
        self.variance: nn.Parameter
        self.register_parameter("variance", nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return self.variance.new_ones([len(x)]) * torch.exp(self.variance * 10.0)

    def __repr__(self) -> str:
        return f"SingleVarianceNetwork(variance: {self.variance.item()})"
