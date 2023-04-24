# Copyright (c) Gorilla-Lab. All rights reserved.
import os
from typing import Optional, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .dataset_base import DatasetBase
from .utils import GtRays, Intrin, Intrin_and_Inv, select_or_shuffle_rays


class ScanNetDataset(DatasetBase):
    focal: float
    c2w: torch.Tensor  # [n_images, 4, 4]
    gt: torch.Tensor  # [n_images, h, w, 3]
    image_ids: torch.Tensor  # [n_images]
    h: int
    w: int
    n_images: int
    rays: Optional[GtRays]
    split: str

    def __init__(
        self,
        data_root: str,
        mvs_root: str,
        scene: str,
        split: str,
        epoch_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        permutation: bool = True,
        factor: int = 1,
        patch_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.split = split
        self.data_root = data_root
        self.scene = scene
        self.instance_dir = os.path.join(data_root, scene)
        assert os.path.exists(self.instance_dir)

        image_dir = os.path.join(self.instance_dir, "images")
        image_list = os.listdir(image_dir)
        image_list.sort(key=lambda x: int(x.split(".")[0]))
        self.n_images = len(image_list)

        self.device = device
        self.permutation = permutation
        self.epoch_size = epoch_size
        self.input_normal = True

        # load the camera parameters and rgb images
        intrinsic_all = []
        pose_all = []
        all_gt = []
        image_ids = []

        self.image_names = []
        intrinsic = np.loadtxt(os.path.join(self.instance_dir, "intrinsic.txt"))[:3, :3]

        if split == "train":
            all_depth_gt = []
            all_planar_mask = []
            all_mvsplanar_normal = []
            clean_suffix = "_clean"
            assert os.path.exists(mvs_root), f"mvs_root: {mvs_root} does not exist"
        
            mvs_mask_suffix = "planar_mask_mvs"
            print(f"Using mvs dir : {mvs_root}")
            print(f"Using planar mask suffix : {mvs_mask_suffix}")

        for idx, imgname in enumerate(
            tqdm(image_list, desc=f"Loading camera parameters and images ^-^ : {split}")
        ):
            self.image_names.append(imgname)
            c2w = np.loadtxt(os.path.join(self.instance_dir, "pose", f"{imgname[:-4]}.txt"))
            pose_all.append(torch.from_numpy(c2w).float())
            intrinsic_all.append(torch.from_numpy(intrinsic).float())

            rgb = imageio.imread(os.path.join(self.instance_dir, "images", f"{imgname[:-4]}.png"))
            all_gt.append(torch.from_numpy(rgb))

            image_ids.append(idx)

            if split == "train":
                depth_path = os.path.join(
                    mvs_root,
                    "planar_prior",
                    f"depth_normal_filter{clean_suffix}",
                    f"{imgname[:-4]}.npy",
                )
                if os.path.exists(depth_path):
                    depth = np.load(depth_path)
                    depth[depth > 4.0] = 0
                else:
                    if self.input_normal:
                        depth = np.zeros((rgb.shape[0], rgb.shape[1], 4), np.float32)
                    else:
                        depth = np.zeros((rgb.shape[0], rgb.shape[1]), np.float32)
                all_depth_gt.append(torch.from_numpy(depth).float())

                planar_mask_path = os.path.join(
                    mvs_root,
                    "..",
                    mvs_mask_suffix,
                    f"{imgname[:-4]}.npy",
                )
                planar_mask = np.load(planar_mask_path)
                all_planar_mask.append(torch.from_numpy(planar_mask))

                mvsplanar_normal_path = os.path.join(
                    mvs_root,
                    "planar_prior",
                    f"planar_normal{clean_suffix}",
                    f"{imgname[:-4]}.npy",
                )
                mvsplanar_normal = np.load(mvsplanar_normal_path)
                all_mvsplanar_normal.append(torch.from_numpy(mvsplanar_normal).float())

        self.c2w = torch.stack(pose_all)
        self.c2w_inv = torch.inverse(self.c2w)
        self.intrins = torch.stack(intrinsic_all)
        self.gt = torch.stack(all_gt).float() / 255.0
        self.image_ids = torch.tensor(image_ids)
        if split == "train":
            if self.input_normal:
                self.depth_gt = torch.stack(all_depth_gt)[..., 0:1]
                self.normal_gt = torch.stack(all_depth_gt)[..., 1:]
            else:
                self.depth_gt = torch.stack(all_depth_gt).unsqueeze(-1)
            self.planar_mask = torch.stack(all_planar_mask)
            self.mvsplanar_normal = torch.stack(all_mvsplanar_normal)
            self.mvsplanar_normal = F.normalize(self.mvsplanar_normal, p=2, dim=-1).clamp(-1.0, 1.0)
            print(f"INFO: valid mvs depths/normals: {(self.depth_gt>0).sum()}")
        self.n_images, self.h_full, self.w_full, _ = self.gt.shape

        self.intrins_full: Intrin = Intrin(
            self.intrins[..., 0, 0],
            self.intrins[..., 1, 1],
            self.intrins[..., 0, 2],
            self.intrins[..., 1, 2],
        )

        self.intrins_and_inv_full: Intrin_and_Inv = Intrin_and_Inv(
            self.intrins, torch.inverse(self.intrins)
        )

        self.split = split
        if self.split == "train":
            self.gen_rays_scannet_neus_keeppatch(
                factor=factor
            ) if patch_size > 1 else self.gen_rays_scannet_neus(factor=factor)
            self.h, self.w = self.h_full, self.w_full
            self.intrins: Intrin = self.intrins_full
        else:
            # Rays are not needed for testing
            self.h, self.w = self.h_full, self.w_full
            self.intrins: Intrin = self.intrins_full

    def shuffle_rays(self, epoch_size: Optional[int] = None) -> Optional[torch.Tensor]:
        """
        Shuffle all rays
        """
        if self.split == "train":
            del self.rays
            self.rays, indexer = select_or_shuffle_rays(
                self.rays_init, self.permutation, epoch_size, self.device
            )
            self.indexer = indexer

    def gen_rays_scannet_neus(self, factor: int = 1) -> None:
        # Generate rays
        self.factor = factor
        self.h = self.h_full // factor
        self.w = self.w_full // factor
        true_factor = self.h_full / self.h
        self.intrins_and_inv = self.intrins_and_inv_full.scale(1.0 / true_factor)

        tx = torch.linspace(0, self.w - 1, self.w)
        ty = torch.linspace(0, self.h - 1, self.h)
        pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing="ij")
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3

        intri_inv = self.intrins_and_inv_full.intrinsic_inv[0]
        p_one = torch.matmul(intri_inv[None, None, :3, :3], p[:, :, :, None]).squeeze()

        depth_ratio = torch.linalg.norm(p_one, ord=2, dim=-1, keepdim=True)
        dirs = p_one / depth_ratio
        dirs = dirs.transpose_(0, 1)
        depth_ratio = depth_ratio.transpose_(0, 1)
        dirs = dirs.reshape(1, -1, 3, 1)
        dirs = (self.c2w[:, None, :3, :3] @ dirs)[..., 0]
        dirs = dirs.contiguous().view(-1, 3)
        origins = self.c2w[:, None, :3, 3].expand(-1, self.h * self.w, -1).contiguous().view(-1, 3)
        gt = self.gt.reshape(self.n_images, -1, 3).contiguous().view(-1, 3)
        # depth_gt = self.depth_gt.reshape(self.n_images, -1, 1)
        depth_gt = (
            (self.depth_gt * depth_ratio[None, ...])
            .reshape(self.n_images, -1, 1)
            .contiguous()
            .view(-1, 1)
        )

        self.rays_init = GtRays(
            origins=origins,
            dirs=dirs,
            gt=gt,
            depth_gt=depth_gt,
            normal_gt=self.normal_gt.view(-1, 3) if self.input_normal else None,
            manhattan_mask=self.planar_mask.view(-1),
            mask=None,
        )
        self.rays = self.rays_init

    def gen_rays_scannet_neus_keeppatch(self, factor: int = 1) -> None:
        assert factor == 1  # note that, we haven't implent the refactor operator
        print("Generate rays in image patchs")
        self.factor = factor
        self.h = self.h_full // factor
        self.w = self.w_full // factor
        true_factor = self.h_full / self.h
        self.intrins_and_inv = self.intrins_and_inv_full.scale(1.0 / true_factor)

        tx = torch.linspace(0, self.w - 1, self.w)
        ty = torch.linspace(0, self.h - 1, self.h)
        pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing="ij")
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3

        intri_inv = self.intrins_and_inv_full.intrinsic_inv[0]
        p_one = torch.matmul(intri_inv[None, None, :3, :3], p[:, :, :, None]).squeeze()

        depth_ratio = torch.linalg.norm(p_one, ord=2, dim=-1, keepdim=True)
        dirs = p_one / depth_ratio
        dirs = dirs.transpose_(0, 1)
        depth_ratio = depth_ratio.transpose_(0, 1)
        dirs = dirs.reshape(1, -1, 3, 1)
        dirs = (self.c2w[:, None, :3, :3] @ dirs)[..., 0]
        dirs = dirs.contiguous().view(-1, self.h, self.w, 3)
        origins = self.c2w[:, None, None, :3, 3].expand(-1, self.h, self.w, -1).contiguous()
        # depth_gt = self.depth_gt
        depth_gt = self.depth_gt * depth_ratio[None, ...]

        self.rays_init = GtRays(
            origins=origins,
            dirs=dirs,
            gt=self.gt,
            depth_gt=depth_gt,
            normal_gt=self.normal_gt if self.input_normal else None,
            manhattan_mask=self.planar_mask,
            mask=None,
        )
        self.rays = self.rays_init
