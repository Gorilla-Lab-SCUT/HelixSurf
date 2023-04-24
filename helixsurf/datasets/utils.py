from dataclasses import dataclass
from typing import Any, List, Optional, Union

import torch


@dataclass
class GtRays:
    origins: Union[torch.Tensor, List[torch.Tensor]]
    dirs: Union[torch.Tensor, List[torch.Tensor]]
    gt: Union[torch.Tensor, List[torch.Tensor]]
    depth_gt: Optional[Union[torch.Tensor, List[torch.Tensor]]]
    normal_gt: Optional[Union[torch.Tensor, List[torch.Tensor]]]
    manhattan_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]]
    mask: Optional[Union[torch.Tensor, List[torch.Tensor]]]

    def to(self, *args, **kwargs) -> "GtRays":
        origins = self.origins.to(*args, **kwargs)
        dirs = self.dirs.to(*args, **kwargs)
        gt = self.gt.to(*args, **kwargs)
        depth_gt = self.depth_gt.to(*args, **kwargs) if self.depth_gt is not None else None
        normal_gt = self.normal_gt.to(*args, **kwargs) if self.normal_gt is not None else None
        manhattan_mask = (
            self.manhattan_mask.to(*args, **kwargs) if self.manhattan_mask is not None else None
        )
        mask = self.mask.to(*args, **kwargs) if self.mask is not None else None
        return GtRays(origins, dirs, gt, depth_gt, normal_gt, manhattan_mask, mask)

    def __getitem__(self, key: str) -> "GtRays":
        origins = self.origins[key]
        dirs = self.dirs[key]
        gt = self.gt[key]
        depth_gt = self.depth_gt[key] if self.depth_gt is not None else None
        normal_gt = self.normal_gt[key] if self.normal_gt is not None else None
        manhattan_mask = self.manhattan_mask[key] if self.manhattan_mask is not None else None
        mask = self.mask[key] if self.mask is not None else None
        return GtRays(origins, dirs, gt, depth_gt, normal_gt, manhattan_mask, mask)

    def __len__(self) -> int:
        return self.origins.size(0)


@dataclass
class GtPretrainedRays:
    origins: Union[torch.Tensor, List[torch.Tensor]]
    dirs: Union[torch.Tensor, List[torch.Tensor]]
    gt: Union[torch.Tensor, List[torch.Tensor]]
    depth_gt: Optional[Union[torch.Tensor, List[torch.Tensor]]]
    normal_gt: Optional[Union[torch.Tensor, List[torch.Tensor]]]
    pretrained_normal: Optional[Union[torch.Tensor, List[torch.Tensor]]]
    pretrained_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]]
    manhattan_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]]
    mask: Optional[Union[torch.Tensor, List[torch.Tensor]]]

    def to(self, *args, **kwargs) -> "GtPretrainedRays":
        origins = self.origins.to(*args, **kwargs)
        dirs = self.dirs.to(*args, **kwargs)
        gt = self.gt.to(*args, **kwargs)
        depth_gt = self.depth_gt.to(*args, **kwargs) if self.depth_gt is not None else None
        normal_gt = self.normal_gt.to(*args, **kwargs) if self.normal_gt is not None else None
        pretrained_normal = (
            self.pretrained_normal.to(*args, **kwargs)
            if self.pretrained_normal is not None
            else None
        )
        pretrained_mask = (
            self.pretrained_mask.to(*args, **kwargs) if self.pretrained_mask is not None else None
        )
        manhattan_mask = (
            self.manhattan_mask.to(*args, **kwargs) if self.manhattan_mask is not None else None
        )
        mask = self.mask.to(*args, **kwargs) if self.mask is not None else None
        return GtPretrainedRays(
            origins,
            dirs,
            gt,
            depth_gt,
            normal_gt,
            pretrained_normal,
            pretrained_mask,
            manhattan_mask,
            mask,
        )

    def __getitem__(self, key: str) -> "GtPretrainedRays":
        origins = self.origins[key]
        dirs = self.dirs[key]
        gt = self.gt[key]
        depth_gt = self.depth_gt[key] if self.depth_gt is not None else None
        normal_gt = self.normal_gt[key] if self.normal_gt is not None else None
        pretrained_normal = (
            self.pretrained_normal[key] if self.pretrained_normal is not None else None
        )
        pretrained_mask = self.pretrained_mask[key] if self.pretrained_mask is not None else None
        manhattan_mask = self.manhattan_mask[key] if self.manhattan_mask is not None else None
        mask = self.mask[key] if self.mask is not None else None
        return GtPretrainedRays(
            origins,
            dirs,
            gt,
            depth_gt,
            normal_gt,
            pretrained_normal,
            pretrained_mask,
            manhattan_mask,
            mask,
        )

    def __len__(self) -> int:
        return self.origins.size(0)


@dataclass
class Intrin:
    fx: Union[float, torch.Tensor]
    fy: Union[float, torch.Tensor]
    cx: Union[float, torch.Tensor]
    cy: Union[float, torch.Tensor]

    def scale(self, scaling: float) -> "Intrin":
        return Intrin(self.fx * scaling, self.fy * scaling, self.cx * scaling, self.cy * scaling)

    def get(self, field: str, image_id: int = 0) -> Any:
        val = self.__dict__[field]
        return val if isinstance(val, float) else val[image_id].item()


@dataclass
class Intrin_and_Inv:
    intrinsic: Union[float, torch.Tensor]
    intrinsic_inv: Union[float, torch.Tensor]

    def scale(self, scaling: float) -> "Intrin_and_Inv":
        return Intrin_and_Inv(
            self.intrinsic[..., :3, :3] * scaling,
            torch.inverse(self.intrinsic[..., :3, :3] * scaling),
        )

    def get(self, field: str, image_id: int = 0) -> Any:
        val = self.__dict__[field]
        return val if isinstance(val, float) else val[image_id].item()


# Data
def select_or_shuffle_rays(
    rays_init: GtRays,
    permutation: bool = False,
    epoch_size: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
) -> GtRays:
    n_rays = rays_init.origins.size(0)
    n_samp = n_rays if (epoch_size is None) else int(epoch_size)
    if permutation and not (n_samp > n_rays):
        print(" Shuffling rays")
        indexer = torch.randperm(n_rays, device="cpu")[:n_samp]
    else:
        print(" Selecting random rays")
        indexer = torch.randint(n_rays, (n_samp,), device="cpu")
    return rays_init[indexer].to(device=device), indexer
