# Copyright (c) Gorilla-Lab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch

from .utils import GtRays, Intrin


class DatasetBase:
    split: str
    permutation: bool
    epoch_size: Optional[int]
    n_images: int
    h_full: int
    w_full: int
    intrins_full: Intrin
    c2w: torch.Tensor  # C2W OpenCV poses
    gt: Union[torch.Tensor, List[torch.Tensor]]  # RGB images
    image_ids: torch.Tensor  # [n_images]
    device: Union[str, torch.device]
    data_root: str

    def __init__(self) -> None:
        self.ndc_coeffs = (-1, -1)
        self.use_sphere_bound = False
        self.should_use_background = True  # a hint
        self.use_sphere_bound = True
        self.scene_center = [0.0, 0.0, 0.0]
        self.scene_radius = [1.0, 1.0, 1.0]
        self.permutation = False

    def get_image_size(self, i: int) -> Tuple[int]:
        if hasattr(self, "image_size"):
            return tuple(self.image_size[i])
        else:
            return self.h, self.w

    def shuffle_rays_withuv(
        self,
        patch_size: int = 1,
        epoch_size: Optional[int] = None,
        shuffle: bool = True,
    ) -> Optional[torch.Tensor]:
        if self.split == "train":
            del self.rays

            assert patch_size > 0
            h_patch_size = patch_size // 2

            # Build index of an image
            B, H, W = self.gt.shape[:3]
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, B - 1, B),
                    torch.linspace(h_patch_size, H - h_patch_size - 1, H - patch_size),
                    torch.linspace(h_patch_size, W - h_patch_size - 1, W - patch_size),
                    indexing="ij",
                ),
                -1,
            ).reshape(-1, 3)
            coords = coords.to(torch.long)

            # Select random rays
            n_rays = len(coords)
            n_samp = n_rays if (epoch_size is None) else int(epoch_size)
            if shuffle:
                if self.permutation and not (n_samp > n_rays):
                    print(f"Shuffle rays with patch uv <--> {patch_size}")
                    indexer = torch.randperm(len(coords), device="cpu")[:n_samp]
                else:
                    print(f"Selecting random rays with patch uv <--> {patch_size}")
                    indexer = torch.randint(len(coords), (n_samp,), device="cpu")
            else:
                indexer = torch.arange(len(coords), device="cpu")[:n_samp]
            choiced_rays_coords = coords[indexer].reshape(-1, 3)
            lb, lh, lw = choiced_rays_coords.unbind(-1)

            self.rays = GtRays(
                origins=self.rays_init.origins[lb, lh, lw].contiguous(),
                dirs=self.rays_init.dirs[lb, lh, lw].contiguous(),
                gt=self.rays_init.gt[lb, lh, lw].contiguous(),
                depth_gt=self.rays_init.depth_gt[lb, lh, lw].contiguous(),
                normal_gt=self.rays_init.normal_gt[lb, lh, lw].contiguous()
                if self.input_normal
                else None,
                manhattan_mask=self.rays_init.manhattan_mask[lb, lh, lw].contiguous(),
                mask=None,
            )
            self.rays = self.rays[...].to(device=self.device)
            self.rays_uv = choiced_rays_coords  # as indexer
