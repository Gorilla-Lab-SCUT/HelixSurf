# Copyright (c) Gorilla-Lab. All rights reserved.
import warnings
from typing import List, Optional

import torch

try:
    import open3d as o3d
except:
    warnings.warn("Can not import open3d, please install")


def bound2cube(
    lower: List[float],
    upper: List[float],
) -> o3d.geometry.LineSet:
    """get the cube lineset of the given bound for drawing
        1 -------- 3
       /|         /|
      5 -------- 7 .
      | |        | |
      . 0 -------- 2
      |/         |/
      4 -------- 6
    Args:
        lower (List[float]): lower bound
        upper (List[float]): upper bound
    Returns:
        o3d.geometry.LineSet: cube lineset
    """
    v0 = [lower[0], lower[1], lower[2]]
    v1 = [lower[0], lower[1], upper[2]]
    v2 = [lower[0], upper[1], lower[2]]
    v3 = [lower[0], upper[1], upper[2]]
    v4 = [upper[0], lower[1], lower[2]]
    v5 = [upper[0], lower[1], upper[2]]
    v6 = [upper[0], upper[1], lower[2]]
    v7 = [upper[0], upper[1], upper[2]]

    points = [v0, v1, v2, v3, v4, v5, v6, v7]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
    ]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    return line_set


def export_valid_grid(
    resolution: int,
    valid_grid: torch.Tensor,
    lower_bound: List[float] = [-1.0, -1.0, -1.0],
    upper_bound: List[float] = [1.0, 1.0, 1.0],
    filename: str = "temp.ply",
    return_lines: bool = False,
) -> Optional[o3d.geometry.LineSet]:
    # call before each epoch to update extra states.
    reso = [(u - l) / resolution for u, l in zip(upper_bound, lower_bound)]

    import open3d as o3d
    from tqdm import tqdm

    from helixsurf.utils.vis import bound2cube

    lines = o3d.geometry.LineSet()
    valid_ids = torch.where(valid_grid)
    x_ids = valid_ids[0].cpu().numpy().tolist()
    y_ids = valid_ids[1].cpu().numpy().tolist()
    z_ids = valid_ids[2].cpu().numpy().tolist()
    for i, j, k in tqdm(zip(x_ids, y_ids, z_ids)):
        lower = [
            lower_bound[0] + i * reso[0],
            lower_bound[1] + j * reso[1],
            lower_bound[2] + k * reso[2],
        ]
        upper = [
            lower_bound[0] + (i + 1) * reso[0],
            lower_bound[1] + (j + 1) * reso[1],
            lower_bound[2] + (k + 1) * reso[2],
        ]
        lines += bound2cube(lower, upper)

    if return_lines:
        return lines
    else:
        o3d.io.write_line_set(filename, lines)
        print(f"save occupancy grid line set as {filename}")
