# Copyright (c) Gorilla-Lab. All rights reserved.
import argparse

from omegaconf import OmegaConf

from .primitives import RenderOptions


def get_default_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # necessary path
    group = parser.add_argument_group("essential path")
    group.add_argument("--data_dir", "-d", type=str, default=None, help="the directory of data")
    group.add_argument(
        "--train_dir",
        "-t",
        type=str,
        default="ckpt",
        help="checkpoint and logging directory",
    )
    group.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Config yaml file (will override args)",
    )

    # data loading
    group = parser.add_argument_group("Data loading")
    group.add_argument(
        "--dataset_type", default="scannet", 
        help="Dataset type Optional: [scannet]"
    )

    # scene data preprocessing options
    group = parser.add_argument_group("scene processing options")
    group.add_argument("--scene", type=str, default="scene0616_00", help="scene to learning")
    group.add_argument("--factor", type=int, default=2, help="downsample factor for LLFF images")

    # rendering
    group = parser.add_argument_group("Render options")
    group.add_argument(
        "--step_size",
        type=float,
        default=0.5,
        help="Render step size (in voxel size units)",
    )
    group.add_argument(
        "--sigma_thresh",
        type=float,
        default=1e-8,
        help="Skips voxels with sigma < this",
    )
    group.add_argument(
        "--stop_thresh", type=float, default=1e-7, help="Ray march stopping threshold"
    )
    group.add_argument(
        "--background_brightness",
        type=float,
        default=0.0,
        help="Brightness of the infinite background",
    )
    group.add_argument(
        "--near_clip",
        type=float,
        default=0.00,
        help="Near clip distance (in world space distance units, only for FG)",
    )
    group.add_argument(
        "--use_spheric_clip",
        action="store_true",
        default=False,
        help="Use spheric ray clipping instead of voxel grid AABB "
        "(only for FG; changes near_clip to mean 1-near_intersection_radius; "
        "far intersection is always at radius 1)",
    )
    group.add_argument(
        "--last_sample_opaque",
        action="store_true",
        default=False,
        help="Last sample has +1e9 density (used for LLFF)",
    )

    # model
    group = parser.add_argument_group("model options")
    group.add_argument("--bound", type=float, default=1.0, help="aabb size.")

    # grid
    group = parser.add_argument_group("Grid options")
    group.add_argument("--grid_res", type=int, default=128, help="grid resolution.")
    group.add_argument("--grid_sample_res", type=int, default=128, help="grid sampling resolution.")
    group.add_argument("--max_grid_res", type=int, default=1024, help="max grid resolution.")
    group.add_argument(
        "--max_grid_sample_res",
        type=int,
        default=1024,
        help="max grid sampling resolution.",
    )
    group.add_argument("--ups_epoch", type=int, default=2, help="grid upsample epoch.")

    # planar normal marching cube
    group = parser.add_argument_group("planar normal marching cube options")
    group.add_argument(
        "--planar_normal_marchingcube_res", type=int, default=128, help="marching cube resolution."
    )
    group.add_argument(
        "--planar_normal_marchingcube_thresh", type=float, default=0.0, help="mraching cube thresh."
    )
    group.add_argument(
        "--plane_slidewindow_size", type=int, default=7, help="marching cube resolution."
    )

    return parser


def merge_config_file(args: argparse.Namespace) -> OmegaConf:
    """
    Load json config file if specified and merge the arguments
    """
    args_conf = OmegaConf.create(vars(args))
    yaml_conf = OmegaConf.load(args.config)
    conf = OmegaConf.merge(args_conf, yaml_conf)
    return conf


def setup_render_opts(opt: RenderOptions, conf: OmegaConf) -> None:
    """
    Pass render arguments to the SparseGrid renderer options
    """
    opt.step_size = conf.step_size
    opt.sigma_thresh = conf.sigma_thresh
    opt.stop_thresh = conf.stop_thresh
    opt.background_brightness = conf.background_brightness
    opt.near_clip = conf.near_clip
    opt.use_spheric_clip = conf.use_spheric_clip
    opt.last_sample_opaque = conf.last_sample_opaque

    # mlp
    opt.bound = conf.bound

    # grid
    opt.grid_res = conf.grid_res
    opt.max_grid_res = conf.max_grid_res
    opt.grid_sample_res = conf.grid_sample_res
    opt.max_grid_sample_res = conf.max_grid_sample_res
