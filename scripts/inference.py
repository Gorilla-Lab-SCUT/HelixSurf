# Copyright (c) Gorilla-Lab. All rights reserved.
import argparse
import math
import os
from typing import Optional, Sequence

import prim3d
import imageio
import numpy as np
import omegaconf
import open3d as o3d
import torch
import torch.nn.functional as F
import trimesh
from tqdm import tqdm

import helixsurf
from helixsurf.utils.mesh import (
    evaluate_mesh,
    o3dmesh_to_trimesh,
    refuse,
    remove_isolate_component_by_diameter,
    transform,
)

os.environ["PYOPENGL_PLATFORM"] = "egl"

helixsurf.set_random_seed(123456)


def get_args() -> argparse.Namespace:
    parser = helixsurf.get_default_args()
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint file to load")
    parser.add_argument(
        "--up_sample_steps",
        type=int,
        default=0,
        help="Number of up sampling. Defaults to 0",
    )
    parser.add_argument(
        "--marching_cube_resolution",
        "--mc_res",
        type=int,
        default=512,
        help="the resolution of marching cubes",
    )
    parser.add_argument(
        "--marching_cube_threshold",
        "--mc_thres",
        type=float,
        default=0,
        help="the density threshold of marching cubes, 0 for SDF",
    )
    parser.add_argument(
        "--mesh_clean_percentage",
        "-mcp",
        type=float,
        default=0.,
        help="percentage to clean the mesh",
    )
    parser.add_argument(
        "--crop_top_ratio",
        type=float,
        default=0.8,
        help="percentage to cut the top of mesh, '1.0' means not crop",
    )
    parser.add_argument(
        "--export_mesh", action="store_true", default=False, help="only export mesh"
    )
    parser.add_argument(
        "--render", action="store_true", default=False, help="rendering results and calculate PSNR"
    )

    args = parser.parse_args()
    return args


@torch.no_grad()
def eval(
    surf: helixsurf.HelixSurf,
    opt: helixsurf.RenderOptions,
    dataset: helixsurf.DatasetBase,
    conf: omegaconf.OmegaConf,
    save_dir: Optional[str] = None,
) -> None:
    """rendering evaluation

    Args:
        surf (helixsurf.HelixSurf): trained model
        opt (helixsurf.RenderOptions): rendering options
        dataset (helixsurf.DatasetBase): dataset
        conf (omegaconf.OmegaConf): configuration
        save_dir (Optional[str], optional): directory to save visualization. Defaults to None.
    """
    # Put in a function to avoid memory leak
    print("evaluation")
    stats_test = {"psnr": 0.0, "mse": 0.0}

    # Standard set
    N_IMGS_TO_EVAL = min(20, dataset.n_images)
    N_IMGS_TO_SAVE = N_IMGS_TO_EVAL  # if not conf.tune_mode else 1
    img_eval_interval = dataset.n_images // N_IMGS_TO_EVAL
    img_save_interval = N_IMGS_TO_EVAL // N_IMGS_TO_SAVE
    img_ids = range(0, dataset.n_images, img_eval_interval)

    n_images_gen = 0
    for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
        height = dataset.get_image_size(img_id)[0]
        width = dataset.get_image_size(img_id)[1]
        c2w = dataset.c2w[img_id].to(device=device)
        cam = helixsurf.Camera(
            c2w,
            dataset.intrins.get("fx", img_id),
            dataset.intrins.get("fy", img_id),
            dataset.intrins.get("cx", img_id),
            dataset.intrins.get("cy", img_id),
            height=height,
            width=width,
            ndc_coeffs=dataset.ndc_coeffs,
        )
        # network forward
        rgb_pred_test, depth_pred_test, normal_pred_test, _ = surf.eval_render(
            opt,
            cam,
            batch_size=4000,
            cos_anneal_ratio=1.0,
            up_sample_steps=conf.up_sample_steps,
        )
        # calculate mse
        rgb_pred_test = rgb_pred_test.clamp(0.0, 1.0)
        rgb_gt_test = dataset.gt[img_id].to(device=device)
        all_mses = ((rgb_gt_test - rgb_pred_test) ** 2).cpu()

        if save_dir and (i % img_save_interval == 0):
            img_pred = rgb_pred_test.cpu()
            img_gt = rgb_gt_test.cpu()
            img_log = torch.cat([img_pred, img_gt], dim=1)
            imageio.imwrite(
                os.path.join(save_dir, f"{img_id}.png"), (img_log * 255).to(torch.uint8)
            )
            # log predicted depth map
            depth_img = helixsurf.utils.viridis_cmap(depth_pred_test.cpu())
            imageio.imwrite(
                os.path.join(save_dir, f"{img_id}_depth.png"),
                (depth_img * 255).astype(np.uint8),
            )
            # log predicted normal map
            normal_pred_test = F.normalize(normal_pred_test, dim=-1, p=2)
            imageio.imwrite(
                os.path.join(save_dir, f"{img_id}_normal.png"),
                np.uint8((normal_pred_test.detach().cpu() + 1) * 127.5),
            )

        mse_num: float = all_mses.mean().item()
        psnr = -10.0 * math.log10(mse_num)
        if math.isnan(psnr):
            print("NAN PSNR", i, img_id, mse_num)
            assert False
        stats_test["mse"] += mse_num
        stats_test["psnr"] += psnr
        n_images_gen += 1

    stats_test["mse"] /= n_images_gen
    stats_test["psnr"] /= n_images_gen

    print("eval stats:", stats_test)

def crop_mesh_top(
    mesh: o3d.geometry.TriangleMesh,
    ratio: float
) -> o3d.geometry.TriangleMesh:
    """crop the top of mesh for convinent visualization

    Args:
        mesh (o3d.geometry.TriangleMesh): input mesh
        ratio (float): z ratio

    Returns:
        o3d.geometry.TriangleMesh: cropped mesh
    """
    pcd = np.array(mesh.vertices)
    min_bound = pcd.min(0)
    max_bound = pcd.max(0)
    z_val = max_bound[2] - min_bound[2]
    new_z = z_val * ratio
    max_bound[2] = new_z + min_bound[2]

    aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    crop_mesh = mesh.crop(aabb)

    return crop_mesh


@torch.no_grad()
def eval_mesh(
    surf: helixsurf.HelixSurf,
    dataset: helixsurf.DatasetBase,
    conf: omegaconf.OmegaConf,
    save_dir: Optional[str] = None,
    scale: float = 1.0,
    offset: Sequence[float] = [0, 0, 0],
    mesh_clean_percent: float = 0.,
) -> None:
    """evaluate reconstructed mesh

    Args:
        surf (helixsurf.HelixSurf): trained model
        dataset (helixsurf.DatasetBase): dataset
        conf (omegaconf.OmegaConf): configuration
        save_dir (Optional[str], optional): directory to saver visualization results. Defaults to None.
        scale (float, optional): transformation parameters. Defaults to 1.0.
        offset (Sequence[float], optional): transformation parameters. Defaults to [0, 0, 0].
        mesh_clean_percent (float, optional): the percent to clean mesh. Defaults to 0..
    """
    resx = int(conf.marching_cube_resolution)
    resy = int(conf.marching_cube_resolution)
    resz = int(conf.marching_cube_resolution)

    bound = conf.bound
    half_grid_size = bound / conf.marching_cube_resolution
    xs = np.linspace(-bound + half_grid_size, bound - half_grid_size, resx)
    ys = np.linspace(-bound + half_grid_size, bound - half_grid_size, resy)
    zs = np.linspace(-bound + half_grid_size, bound - half_grid_size, resz)
    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
    samplepos = np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=-1)
    samplepos = torch.from_numpy(samplepos).float().cuda()

    batch_size = 720720
    all_sdfgrid = torch.Tensor([]).cuda()
    for i in tqdm(range(0, len(samplepos), batch_size)):
        with torch.no_grad():
            sample_vals_sdf = surf.sdf_net.sdf(samplepos[i : i + batch_size])
        # all_sdfgrid.append(sample_vals_sdf)
        all_sdfgrid = torch.cat([all_sdfgrid, sample_vals_sdf])
        del sample_vals_sdf
        torch.cuda.empty_cache()

    # sdfgrid = torch.cat(all_sdfgrid, dim=0).view(resx, resy, resz)
    sdfgrid = all_sdfgrid.view(resx, resy, resz)
    sdfgrid = sdfgrid.reshape(resx, resy, resz)

    vertices, faces = prim3d.marching_cubes(
        sdfgrid,
        float(conf.marching_cube_threshold),
        ([-bound] * 3, [bound] * 3),
        verbose=False,
    )

    tri_mesh = trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy())

    pred_mesh = refuse(tri_mesh, dataset, scale)
    if save_dir:
        o3d_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(tri_mesh.vertices),
            triangles=o3d.utility.Vector3iVector(tri_mesh.faces))
        crop_o3d_mesh = crop_mesh_top(o3d_mesh, conf.crop_top_ratio)
        o3d.io.write_triangle_mesh(
            os.path.join(save_dir, f"predicted_raw.ply"), crop_o3d_mesh
        )
        crop_pred_mesh = crop_mesh_top(pred_mesh, conf.crop_top_ratio)
        o3d.io.write_triangle_mesh(
            os.path.join(save_dir, f"predicted.ply"), crop_pred_mesh
        )

    pred_mesh = transform(pred_mesh, scale, offset)
    if save_dir:
        crop_pred_mesh = crop_mesh_top(pred_mesh, conf.crop_top_ratio)
        o3d.io.write_triangle_mesh(os.path.join(save_dir, f"predicted_transform.ply"), crop_pred_mesh)

    mesh_gt = o3d.io.read_triangle_mesh(
        # os.path.join(dataset.data_root, dataset.scene, f"{dataset.scene}_rotgt_clean.ply")
        os.path.join(dataset.data_root, dataset.scene, f"{dataset.scene}_manhattansdf.obj")
    )
    evaluate_result = evaluate_mesh(pred_mesh, mesh_gt)
    for k, v in evaluate_result.items():
        print(f"{k:7s}: {v:1.3f}")

    if mesh_clean_percent > 0.001:
        # Eval after cleaning
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices.cpu().numpy())
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
        o3d_mesh = remove_isolate_component_by_diameter(o3d_mesh, mesh_clean_percent)

        tri_mesh = o3dmesh_to_trimesh(o3d_mesh)
        pred_mesh = refuse(tri_mesh, dataset, scale)
        if save_dir:
            crop_o3d_mesh = crop_mesh_top(o3d_mesh, conf.crop_top_ratio)
            o3d.io.write_triangle_mesh(
                os.path.join(save_dir, f"predicted_raw_cleaned.ply"), crop_o3d_mesh
            )
            crop_pred_mesh = crop_mesh_top(pred_mesh, conf.crop_top_ratio)
            o3d.io.write_triangle_mesh(
                os.path.join(save_dir, f"predicted_cleaned.ply"), crop_pred_mesh
            )

        pred_mesh = transform(pred_mesh, scale, offset)

        evaluate_result = evaluate_mesh(pred_mesh, mesh_gt)
        print(f"Score after cleaning the predicted mesh with diameter percentage {mesh_clean_percent} : ")
        for k, v in evaluate_result.items():
            print(f"{k:7s}: {v:1.3f}")


if __name__ == "__main__":
    # prase args
    args = get_args()
    conf = helixsurf.merge_config_file(args)
    print("Config:\n", omegaconf.OmegaConf.to_yaml(conf))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # update the model parameters
    conf.model.bound = conf.bound
    conf.model.grid_res = conf.grid_res
    surf = helixsurf.HelixSurf(**conf.model).to(device)
    surf.eval()
    checkpoint = helixsurf.load_checkpoint(surf, conf.ckpt, strict=True)

    # load normalize matrix to get the origin size
    trans_file = os.path.join(conf.data_dir, conf.scene, "trans_n2w.txt")
    if os.path.exists(trans_file):
        trans_n2w = np.loadtxt(trans_file)
        scale = trans_n2w[0, 0]
        offset = trans_n2w[:3, 3]
    else:
        scale = 1.0
        offset = np.array([0., 0., 0.])
    print(f"mesh scale {scale} | mesh offsets {offset}")

    if conf.export_mesh:
        print("begin marching cubes to export mesh")
        surf.export_mesh(
            filename=conf.ckpt.replace(".pth", ".ply"),
            resolution=conf.marching_cube_resolution,
            batch_size=720720,
        )
        norm_mesh = o3d.io.read_triangle_mesh(conf.ckpt.replace(".pth", ".ply"))
        if float(conf.mesh_clean_percentage)>0.001:
            clean_mesh = remove_isolate_component_by_diameter(
                norm_mesh,
                float(conf.mesh_clean_percentage),
                keep_mesh=False
            )
            o3d.io.write_triangle_mesh(conf.ckpt.replace(".pth", "_clean.ply"), clean_mesh)
        if scale != 1 and np.abs(offset).sum() != 0. :
            pred_mesh = transform(norm_mesh, scale, offset)
            o3d.io.write_triangle_mesh(conf.ckpt.replace(".pth", "_world.ply"), pred_mesh)

    else:
        # set render options
        opt = helixsurf.RenderOptions()
        helixsurf.setup_render_opts(opt, conf)
        print("Render options:\n", opt)

        # init dataset
        factor = 1
        test_dataset = helixsurf.datasets[conf.dataset_type](
            conf.data_dir,
            mvs_root=None,
            scene=conf.scene,
            split="test",
            device=device,
            factor=factor,
        )

        # evaluation
        if conf.render:
            save_dir = os.path.join(os.path.dirname(conf.ckpt), "rendering")
            os.makedirs(save_dir, exist_ok=True)
            eval(surf, opt, test_dataset, conf, save_dir)
        else:
            save_dir = os.path.dirname(conf.ckpt)
            eval_mesh(surf, 
                test_dataset, 
                conf, 
                save_dir, 
                scale,
                offset, 
                float(conf.mesh_clean_percentage),
            )
