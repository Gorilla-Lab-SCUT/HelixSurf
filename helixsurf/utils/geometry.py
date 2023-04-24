# Copyright (c) Gorilla-Lab. All rights reserved.
from typing import Optional, Tuple

import torch
import torch.nn as nn


def convert_to_ndc(origins, directions, ndc_coeffs, near: float = 1.0):
    """Convert a set of rays to NDC coordinates."""
    # Shift ray origins to near plane, not sure if needed
    t = (near - origins[Ellipsis, 2]) / directions[Ellipsis, 2]
    origins = origins + t[Ellipsis, None] * directions

    dx, dy, dz = directions.unbind(-1)
    ox, oy, oz = origins.unbind(-1)

    # Projection
    o0 = ndc_coeffs[0] * (ox / oz)
    o1 = ndc_coeffs[1] * (oy / oz)
    o2 = 1 - 2 * near / oz

    d0 = ndc_coeffs[0] * (dx / dz - ox / oz)
    d1 = ndc_coeffs[1] * (dy / dz - oy / oz)
    d2 = 2 * near / oz

    origins = torch.stack([o0, o1, o2], -1)
    directions = torch.stack([d0, d1, d2], -1)
    return origins, directions


def normalize_grid(
    flow: torch.Tensor, h: int, w: int, clamp: Optional[float] = None
) -> torch.Tensor:
    # either h and w are simple float or N torch.tensor where N batch size
    try:
        h.device
    except AttributeError:
        h = torch.tensor(h, device=flow.device).float().unsqueeze(0)
        w = torch.tensor(w, device=flow.device).float().unsqueeze(0)

    if len(flow.shape) == 4:
        w = w.unsqueeze(1).unsqueeze(2)
        h = h.unsqueeze(1).unsqueeze(2)
    elif len(flow.shape) == 3:
        w = w.unsqueeze(1)
        h = h.unsqueeze(1)
    elif len(flow.shape) == 5:
        w = w.unsqueeze(0).unsqueeze(2).unsqueeze(2)
        h = h.unsqueeze(0).unsqueeze(2).unsqueeze(2)

    res = torch.empty_like(flow)
    if res.shape[-1] == 3:
        res[..., 2] = 1

    # for grid_sample with align_corners=True
    # https://github.com/pytorch/pytorch/blob/c371542efc31b1abfe6f388042aa3ab0cef935f2/aten/src/ATen/native/GridSampler.h#L33
    res[..., 0] = 2 * flow[..., 0] / (w - 1) - 1
    res[..., 1] = 2 * flow[..., 1] / (h - 1) - 1

    if clamp:
        return torch.clamp(res, -clamp, clamp)
    else:
        return res


# align_smooth_norm from https://github.com/SJTU-ViSYS/StructDepth/blob/main/layers.py
def compute_manhattan_gt(
    norm: torch.Tensor,  # [n_rays, 3]
    manhattan_norm: torch.Tensor,  # [n_rays, 6, 3]
    thresh: float = 0.9
) -> Tuple[torch.Tensor]:
    n_rays = manhattan_norm.shape[0]
    norm = norm[:, None, :].expand(n_rays, 6, 3)  # [n_rays, 6, 3]
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    cos_sim = cos(manhattan_norm, norm)  # [n_rays, 6]
    score_map, mmap = torch.max(cos_sim, 1)  # [n_rays]

    mmap_label = mmap[:, None, None].expand(n_rays, 3, 1)  # [n_rays, 3, 1]
    manhattan_norm = manhattan_norm.permute(0, 2, 1)  # [n_rays, 3, 6]
    aligned_manhattan_norm = torch.gather(manhattan_norm, 2, mmap_label).squeeze()  # [n_rays, 3]

    # The mask here first comes from the top edge and the right edge in depth2norm.
    mmap_mask = torch.ones_like(mmap)

    # When the estimated normal is very close to the given principal direction, \
    # NaN with cos greater than 1 will appear, so NaN will be set to 1 here.
    if torch.any(torch.isnan(score_map)):
        print("nan in mmap compute! set nan = 1")
        torch.nan_to_num(score_map, nan=1)

    # Secondly, an adaptive threshold is used to filter the pixels with too large an Angle deviation,\
    # with an initial Angle of about 25 degrees (thresh=0.9)
    mmap_mask[score_map < thresh] = 0
    mmap_mask = mmap_mask.bool()

    return aligned_manhattan_norm, mmap_mask
