# Copyright (c) Gorilla-Lab. All rights reserved.
import argparse
import os

import cv2
import numpy as np
import omegaconf
import open3d as o3d
import prim3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import helixsurf
import helixsurf.libvolume as _C
from helixsurf.utils.mesh import remove_isolate_component_by_diameter
from helixsurf.utils.misc import k_means, fast_k_means, viridis_cmap

helixsurf.set_random_seed(123456)

"""
NOTE:
    in the dataset preprocess stage, we capture the manhattan directions
    (the gravity axis and the major horizontal axis of the Manhattan)
    using COLMAP-sfm (model_orientation_aligner). (https://colmap.github.io/faq.html#manhattan-world-alignment)
    Then we rotate the camera posed using these captured manhattan directions, which means that
    we can set three axis as x = [1, 0, 0], y = [0, 1, 0] and z = [0, 0, 1] directly.
"""
MANHATTAN_AXES = (
    torch.from_numpy(
        np.array(
            [
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1],
            ]
        )
    )
    .float()
    .to("cuda" if torch.cuda.is_available() else "cpu")
)


def get_args() -> argparse.Namespace:
    parser = helixsurf.get_default_args()

    parser.add_argument("--load_ckpt", type=str, required=True, default=None)
    parser.add_argument(
        "--mvs_dir",
        type=str,
        help="directory to the mvs",
    )
    # ray casting marching cube
    parser.add_argument(
        "--ray_casting_marchingcube_res",
        type=int,
        default=256,
        help="marching cube resolution.",
    )
    parser.add_argument(
        "--ray_casting_marchingcube_thresh",
        type=float,
        default=0.0,
        help="mraching cube thresh.",
    )

    args = parser.parse_args()
    return args


def kmeans_plane(
    input_smooth_normal: torch.Tensor,  # [H, W, 3]
    planar_seg_mask: torch.Tensor,  # [H, W], seg=0 for background
    var_thresh: float = 0.9,
    filter_twoplane_thresh: float = 0.1,
    filter_thrplane_thresh: float = 0.25,  # if set number large than 1, means no 3 plane
    fast_kmeans: bool = False,
    simil_fun=None,
):
    # consistency the normal with planes in one image
    seg_plane_normal = torch.zeros_like(input_smooth_normal)
    for seg in planar_seg_mask.unique()[1:]:  # jump 0
        seg_normal = input_smooth_normal[planar_seg_mask == seg]
        mean_dir = seg_normal.mean(0)
        cos_sim = simil_fun(seg_normal, mean_dir)
        cos_score = (cos_sim < var_thresh).sum() / len(seg_normal)
        # oversegmante the planes again, to avoid more than one planes in one seg
        """ two planes in one seg """
        if cos_score > filter_twoplane_thresh:
            centroids, labels = fast_k_means(seg_normal, k=2) \
                if fast_kmeans else k_means(seg_normal, k=2)
            if len(labels.unique()) == 1:
                import ipdb

                ipdb.set_trace()
            cent1_normal = seg_normal[labels == 0]
            cent2_normal = seg_normal[labels == 1]
            cos_sim_1 = simil_fun(cent1_normal, centroids[0])
            cos_sim_2 = simil_fun(cent2_normal, centroids[1])
            cos_score1 = (cos_sim_1 < var_thresh).sum() / (labels == 0).sum()
            cos_score2 = (cos_sim_2 < var_thresh).sum() / (labels == 1).sum()
            """ three planes in one seg """
            if max(cos_score1, cos_score2) > filter_thrplane_thresh:
                centroids3, labels3 = fast_k_means(seg_normal, k=3) \
                    if fast_kmeans else k_means(seg_normal, k=3)
                if len(centroids3) == 3:
                    temp = torch.zeros_like(seg_normal)
                    temp[labels3 == 0] = centroids3[0]
                    temp[labels3 == 1] = centroids3[1]
                    temp[labels3 == 2] = centroids3[2]
                    seg_plane_normal[planar_seg_mask == seg] = temp
                else:
                    temp = torch.where(
                        labels[..., None].expand(-1, 3) > 0.5, centroids[1], centroids[0]
                    )
                    seg_plane_normal[planar_seg_mask == seg] = temp
            else:
                temp = torch.where(
                    labels[..., None].expand(-1, 3) > 0.5, centroids[1], centroids[0]
                )
                seg_plane_normal[planar_seg_mask == seg] = temp
        else:  # one plane
            seg_plane_normal[planar_seg_mask == seg] = mean_dir

    seg_plane_normal = F.normalize(seg_plane_normal, p=2, dim=-1).clamp(-1.0, 1.0)
    return seg_plane_normal


@torch.no_grad()
def generate_normal_plane(
    surf: helixsurf.HelixSurf,
    conf: omegaconf.OmegaConf,
    dataset: helixsurf.ScanNetDataset,
    mvs_weight: float = 1.0,
    plane_slidewindow_size: int = 31,
    clean_mesh: bool = False,
    clean_percent: float = 0.06,
    save_mesh: str = None,
    consistant: bool = False,
    vis: bool = False,
    vis_dir: str = None,
) -> None:
    print(
        f"acquire planar normals type 1 with plane sliding window size <++> {plane_slidewindow_size}"
    )
    H, W = dataset.get_image_size(0)

    # marching cubes to get the shape
    resx = int(conf.planar_normal_marchingcube_res)
    resy = int(conf.planar_normal_marchingcube_res)
    resz = int(conf.planar_normal_marchingcube_res)

    bound = conf.bound
    half_grid_size = bound / conf.planar_normal_marchingcube_res
    xs = np.linspace(-bound + half_grid_size, bound - half_grid_size, resx)
    ys = np.linspace(-bound + half_grid_size, bound - half_grid_size, resy)
    zs = np.linspace(-bound + half_grid_size, bound - half_grid_size, resz)
    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
    samplepos = np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=-1)
    samplepos = torch.from_numpy(samplepos).float().cuda()

    batch_size = 720720
    all_sdfgrid = torch.Tensor([]).cuda()
    for i in tqdm(range(0, len(samplepos), batch_size)):
        sample_vals_sdf = surf.sdf_net.sdf(samplepos[i : i + batch_size])
        all_sdfgrid = torch.cat([all_sdfgrid, sample_vals_sdf])
        del sample_vals_sdf
        torch.cuda.empty_cache()

    sdfgrid = all_sdfgrid.view(resx, resy, resz)
    sdfgrid = sdfgrid.reshape(resx, resy, resz)

    vertices, faces = prim3d.marching_cubes(
        sdfgrid,
        float(conf.planar_normal_marchingcube_thresh),
        ([-bound] * 3, [bound] * 3),
        verbose=False,
    )
    if save_mesh is not None:
        os.makedirs(os.path.dirname(save_mesh), exist_ok=True)
        prim3d.save_mesh(vertices, faces, filename=save_mesh)

    # clean the mesh
    if clean_mesh:
        vertices = vertices.cpu().numpy()
        faces = faces.cpu().numpy()
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
        o3d_mesh = remove_isolate_component_by_diameter(o3d_mesh, clean_percent)
        if save_mesh is not None:
            o3d.io.write_triangle_mesh(f"{save_mesh[:-4]}_cleaned{save_mesh[-4:]}", o3d_mesh)
        vertices = np.asarray(o3d_mesh.vertices)
        faces = np.asarray(o3d_mesh.triangles)
        vertices_rc = torch.from_numpy(vertices).float().cuda()
        faces_rc = torch.from_numpy(faces).to(torch.int32).cuda()
    else:
        vertices_rc = vertices.float()
        faces_rc = faces.to(torch.int32)

    RT = prim3d.libPrim3D.create_raycaster(vertices_rc, faces_rc)
    torch.cuda.empty_cache()

    all_planar_normal = torch.zeros_like(dataset.mvsplanar_normal)
    primitives_normals = torch.zeros([faces.shape[0], 3], dtype=torch.float).cuda()
    primitives_counts = torch.zeros([faces.shape[0]], dtype=torch.int32).cuda()
    primitive_ids_dict = {}

    # -------------------------------------
    # save for vis and debug
    if vis:
        all_plane_normal = {}
        all_smooth_normal = {}
        all_segs_plane_normal = {}

    # -------------------------------------
    dataset.c2w = dataset.c2w.cuda()  # move cam to cuda
    cos = nn.CosineSimilarity()  # function to calculate similarity
    for img_id in tqdm(range(dataset.n_images), total=dataset.n_images):
        planar_mask = dataset.planar_mask[img_id].cuda()
        if (planar_mask == 0).all():
            smooth_normal = torch.zeros((H, W, 3), dtype=torch.float, device="cuda")
            all_planar_normal[img_id] = smooth_normal
            continue

        # find the planar pixel
        planar_region = planar_mask > 0
        # get the rays(origins+dirs) of the valid_planar_mask pixels
        c2w = dataset.c2w[img_id]
        cam = helixsurf.Camera(
            c2w,
            dataset.intrins.get("fx", img_id),
            dataset.intrins.get("fy", img_id),
            dataset.intrins.get("cx", img_id),
            dataset.intrins.get("cy", img_id),
            height=H,
            width=W,
            ndc_coeffs=dataset.ndc_coeffs,
        )
        rays, _ = cam.gen_rays(px_center=0.0, norm_dir=False)
        planar_rays = rays[planar_region.reshape(-1)]
        rays_origins = planar_rays.origins
        rays_dirs = planar_rays.dirs
        assert rays_origins.shape == rays_dirs.shape
        num_rays = rays_origins.shape[0]
        # do ray casting for mesh
        normals = torch.zeros_like(rays_origins)
        depth = torch.zeros([num_rays], dtype=torch.float32, device="cuda")
        primitive_ids = torch.zeros([num_rays], dtype=torch.int32, device="cuda") - 1
        RT.invoke(rays_origins, rays_dirs, depth, normals, primitive_ids)
        normals = F.normalize(normals, p=2, dim=-1).clamp(-1.0, 1.0)

        # multi-view consistant
        primitive_ids += 1  # NOTE: add 1 here
        primitive_ids_map = torch.zeros([H, W], dtype=torch.int32, device="cuda")
        primitive_ids_map[torch.where(planar_region)] = primitive_ids
        primitive_ids_dict[img_id] = primitive_ids_map

        # fusion the mvs normal and ray casting normal for the planar superpixel
        plane_normal = torch.zeros([H, W, 3], dtype=torch.float32, device="cuda")
        if mvs_weight > 0.0:
            # get mvs normal
            normal_mvs = dataset.mvsplanar_normal[img_id].cuda()
            mvs_weight_mask = torch.zeros_like(normal_mvs)
            mvs_weight_mask[torch.where(normal_mvs.abs().sum(-1) > 0.5)] = mvs_weight
            plane_normal[torch.where(planar_region)] = (
                mvs_weight_mask[torch.where(planar_region)] * normal_mvs[torch.where(planar_region)]
                + (1.0 - mvs_weight_mask[torch.where(planar_region)]) * normals
            )
            plane_normal = F.normalize(plane_normal, p=2, dim=-1).clamp(-1.0, 1.0)
        else:
            plane_normal[torch.where(planar_region)] = normals

        # get the smooth normal
        smooth_normal = _C.sliding_window_normal_cu(
            planar_region, plane_normal, plane_slidewindow_size, int(plane_slidewindow_size / 2)
        )
        smooth_normal = F.normalize(smooth_normal, p=2, dim=-1).clamp(-1.0, 1.0)
        # adaptive k-means clustering
        seg_plane_normal = kmeans_plane(smooth_normal, planar_mask, simil_fun=cos)

        # ------------------------------------------------------
        if vis:
            all_plane_normal[img_id] = plane_normal
            all_smooth_normal[img_id] = smooth_normal
            all_segs_plane_normal[img_id] = seg_plane_normal
        # ------------------------------------------------------
        if consistant:
            # consist all seg plane normals with primitive
            _C.count_primitives_cu(
                primitive_ids_map, seg_plane_normal, primitives_normals, primitives_counts
            )
        else:
            all_planar_normal[img_id] = seg_plane_normal

    # backproj the normals on primitives to each image
    if consistant:
        primitives_counts[primitives_counts == 0] = 1
        primitives_normals /= primitives_counts[:, None]
        primitives_normals = F.normalize(primitives_normals, p=2, dim=-1).clamp(-1.0, 1.0)
        for img_id in tqdm(range(dataset.n_images), total=dataset.n_images):
            if img_id not in primitive_ids_dict:
                continue
            # find the planar pixel
            planar_mask = dataset.planar_mask[img_id].cuda()
            planar_region = planar_mask > 0
            # get the consistency smooth normal for each img
            consistent_segplane_normal = torch.zeros([H, W, 3], dtype=torch.float32, device="cuda")
            primitive_ids = primitive_ids_dict[img_id].long()
            consistent_segplane_normal[planar_region] = primitives_normals[
                primitive_ids[planar_region] - 1
            ]  # NOTE: minus 1 here

            all_planar_normal[img_id] = consistent_segplane_normal

            if vis:
                consistent_segplane_normal_rgb = (
                    ((consistent_segplane_normal + 1) * 127.5).cpu().numpy().astype(np.uint8)
                )
                color_bar = np.random.randint(0, 255, [planar_mask.cpu().numpy().max() + 1, 3])
                color_bar[0] = 0
                rgb = (dataset.gt[img_id].cpu().numpy() * 255).astype(np.uint8)
                rgb = rgb[..., ::-1]
                normal_mvs = dataset.mvsplanar_normal[img_id]
                normal_mvs_rgb = ((normal_mvs.cpu().numpy() + 1) * 127.5).astype(np.uint8)
                planar_mask_rgb = color_bar[planar_mask.cpu().numpy().astype(np.int32)]
                # plane normal
                plane_normal = all_plane_normal[img_id]
                plane_normal_rgb = ((plane_normal + 1) * 127.5).cpu().numpy().astype(np.uint8)
                # smooth normal
                smooth_normal = all_smooth_normal[img_id]
                smooth_normal_rgb = ((smooth_normal + 1) * 127.5).cpu().numpy().astype(np.uint8)
                # segplane normal
                seg_plane_normal = all_segs_plane_normal[img_id]
                seg_plane_normal_rgb = (
                    ((seg_plane_normal + 1) * 127.5).cpu().numpy().astype(np.uint8)
                )

                output_img = np.concatenate(
                    [
                        np.concatenate([rgb, normal_mvs_rgb, planar_mask_rgb], axis=1),
                        np.concatenate(
                            [
                                plane_normal_rgb,
                                seg_plane_normal_rgb,
                                consistent_segplane_normal_rgb,
                            ],
                            axis=1,
                        ),
                    ],
                    axis=0,
                )
                vis_dir = "temp_planenormal_vis" if vis_dir is None else vis_dir
                os.makedirs(vis_dir, exist_ok=True)
                vis_path = os.path.join(f"{vis_dir}/{int(img_id):04d}.png")
                cv2.imwrite(vis_path, output_img)
    elif not consistant and vis:
        for img_id in tqdm(range(dataset.n_images), total=dataset.n_images):
            if img_id not in all_smooth_normal:
                continue
            # find the planar pixel
            planar_mask = dataset.planar_mask[img_id]
            planar_region = planar_mask > 0
            normal_mvs = dataset.mvsplanar_normal[img_id]
            """ vis """
            color_bar = np.random.randint(0, 255, [planar_mask.max() + 1, 3])
            color_bar[0] = 0
            rgb = (dataset.gt[img_id].cpu().numpy() * 255).astype(np.uint8)
            rgb = rgb[..., ::-1]  # imageio rgb to cv2 bgr

            normal_mvs_rgb = ((normal_mvs.cpu().numpy() + 1) * 127.5).astype(np.uint8)
            planar_mask_rgb = color_bar[planar_mask.cpu().numpy().astype(np.int32)]

            # plane normal
            plane_normal = all_plane_normal[img_id]
            plane_normal_rgb = ((plane_normal + 1) * 127.5).cpu().numpy().astype(np.uint8)
            # smooth normal
            smooth_normal = all_smooth_normal[img_id]
            smooth_normal_rgb = ((smooth_normal + 1) * 127.5).cpu().numpy().astype(np.uint8)
            # final normal
            final_normal = all_planar_normal[img_id]
            final_normal_rgb = ((final_normal + 1) * 127.5).cpu().numpy().astype(np.uint8)

            output_img = np.concatenate(
                [
                    np.concatenate(
                        [rgb, normal_mvs_rgb, planar_mask_rgb],
                        axis=1,
                    ),
                    np.concatenate(
                        [plane_normal_rgb, smooth_normal_rgb, final_normal_rgb],
                        axis=1,
                    ),
                ],
                axis=0,
            )
            vis_dir = "temp_planenormal_vis" if vis_dir is None else vis_dir
            os.makedirs(vis_dir, exist_ok=True)
            vis_path = os.path.join(f"{vis_dir}/{int(img_id):04d}.png")
            cv2.imwrite(vis_path, output_img)
    else:
        pass

    del RT
    return all_planar_normal


@torch.no_grad()
def ray_casting_depth_normal(
    surf: helixsurf.HelixSurf,
    conf: omegaconf.OmegaConf,
    dataset: helixsurf.ScanNetDataset,
    disturb_depth_noise: float = 0.3,
    disturb_normal_noise: float = 0.3,
    clean_mesh: bool = True,
    clean_percent: float = 0.1,
    save_dir: str = None,
    save_mesh: str = None,
    vis: bool = False,
    vis_dir: str = None,
) -> None:
    print("ray casting depth and normals")
    H, W = dataset.get_image_size(0)

    # marching cubes to get the shape
    resx = int(conf.ray_casting_marchingcube_res)
    resy = int(conf.ray_casting_marchingcube_res)
    resz = int(conf.ray_casting_marchingcube_res)

    bound = conf.bound
    half_grid_size = bound / conf.ray_casting_marchingcube_res
    xs = np.linspace(-bound + half_grid_size, bound - half_grid_size, resx)
    ys = np.linspace(-bound + half_grid_size, bound - half_grid_size, resy)
    zs = np.linspace(-bound + half_grid_size, bound - half_grid_size, resz)
    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
    samplepos = np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=-1)
    samplepos = torch.from_numpy(samplepos).float().cuda()

    batch_size = 720720
    all_sdfgrid = torch.Tensor([]).cuda()
    for i in tqdm(range(0, len(samplepos), batch_size)):
        sample_vals_sdf = surf.sdf_net.sdf(samplepos[i : i + batch_size])
        all_sdfgrid = torch.cat([all_sdfgrid, sample_vals_sdf])
        del sample_vals_sdf
        torch.cuda.empty_cache()

    sdfgrid = all_sdfgrid.view(resx, resy, resz)
    sdfgrid = sdfgrid.reshape(resx, resy, resz)

    vertices, faces = prim3d.marching_cubes(
        sdfgrid,
        float(conf.ray_casting_marchingcube_thresh),
        ([-bound] * 3, [bound] * 3),
        verbose=False,
    )
    if save_mesh is not None:
        os.makedirs(os.path.dirname(save_mesh), exist_ok=True)
        prim3d.save_mesh(vertices, faces, filename=save_mesh)

    # clean the mesh
    if clean_mesh:
        vertices = vertices.cpu().numpy()
        faces = faces.cpu().numpy()
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
        o3d_mesh = remove_isolate_component_by_diameter(o3d_mesh, clean_percent)
        if save_mesh is not None:
            o3d.io.write_triangle_mesh(f"{save_mesh[:-4]}_cleaned{save_mesh[-4:]}", o3d_mesh)
        vertices = np.asarray(o3d_mesh.vertices)
        faces = np.asarray(o3d_mesh.triangles)
        vertices_rc = torch.from_numpy(vertices).float().cuda()
        faces_rc = torch.from_numpy(faces).to(torch.int32).cuda()
    else:
        vertices_rc = vertices.float()
        faces_rc = faces.to(torch.int32)

    RT = prim3d.libPrim3D.create_raycaster(vertices_rc, faces_rc)
    torch.cuda.empty_cache()

    dataset.c2w = dataset.c2w.cuda()
    for img_id in tqdm(range(dataset.n_images), total=dataset.n_images):
        # get the rays(origins+dirs)
        c2w = dataset.c2w[img_id]
        cam = helixsurf.Camera(
            c2w,
            dataset.intrins.get("fx", img_id),
            dataset.intrins.get("fy", img_id),
            dataset.intrins.get("cx", img_id),
            dataset.intrins.get("cy", img_id),
            height=H,
            width=W,
            ndc_coeffs=dataset.ndc_coeffs,
        )
        # caution: the ray dir should not be normalized if ray casting !!!
        rays, _ = cam.gen_rays(px_center=0.0, norm_dir=False)
        rays_origins = rays.origins.view(-1, 3)
        rays_dirs = rays.dirs.view(-1, 3)
        num_rays = rays_origins.shape[0]

        normal = torch.zeros_like(rays_origins)
        depth = torch.zeros([num_rays], dtype=torch.float32, device="cuda")
        primitive_ids = torch.zeros([num_rays], dtype=torch.int32, device="cuda") - 1
        RT.invoke(rays_origins, rays_dirs, depth, normal, primitive_ids)
        normal = F.normalize(normal, p=2, dim=-1).clamp(-1.0, 1.0)

        normal = normal.cpu().numpy().reshape(H, W, 3)
        depth = depth.cpu().numpy().reshape(H, W)

        # disturbe the depth and normal
        depth += np.random.uniform(
            low=-disturb_depth_noise, high=disturb_depth_noise + 0.001, size=depth.shape
        )
        normal += np.random.uniform(
            low=-disturb_normal_noise, high=disturb_normal_noise + 0.001, size=normal.shape
        )

        normal_norm = np.linalg.norm(normal, ord=2, axis=-1, keepdims=True)
        normal_norm = np.where(normal_norm > 0.5, normal_norm, 1.0)  # safety divide
        normal = (normal / normal_norm).clip(-1.0, 1.0)

        save_path = os.path.join(save_dir, f"{int(img_id):04d}")
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, "depth.npy"), depth.astype(np.float32))
        np.save(os.path.join(save_path, "normal.npy"), normal.astype(np.float32))
        np.save(os.path.join(save_path, "cost.npy"), np.zeros_like(depth))

        """ vis """
        if vis:
            rgb = (dataset.gt[img_id].cpu().numpy() * 255).astype(np.uint8)
            rgb = rgb[..., ::-1]
            normal_rgb = ((normal + 1) * 127.5).astype(np.uint8)
            depth_rgb = (viridis_cmap(depth) * 255).astype(np.uint8)
            output_img = np.concatenate([rgb, normal_rgb, depth_rgb], axis=1)
            vis_dir = "temp_visdepthnormal" if vis_dir is None else vis_dir
            os.makedirs(vis_dir, exist_ok=True)
            vis_path = os.path.join(f"{vis_dir}/{int(img_id):04d}.png")
            cv2.imwrite(vis_path, output_img)
    # release the scene
    del RT


if __name__ == "__main__":
    # prase args
    args = get_args()
    conf = helixsurf.merge_config_file(args)
    print("Config:\n", omegaconf.OmegaConf.to_yaml(conf))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    conf.model.bound = conf.bound
    conf.model.grid_res = conf.grid_res
    surf = helixsurf.HelixSurf(**conf.model).to(device)
    surf.eval()

    # init dataset
    factor = 1
    dataset: helixsurf.ScanNetDataset = helixsurf.datasets[conf.dataset_type](
        conf.data_dir,
        mvs_root=conf.mvs_dir,
        scene=conf.scene,
        split="train",
        device=device,
        factor=factor,
        patch_size=1,
    )

    # use the view frustums of cameras to skip the empty grids(useless for nerf-syn)
    ckpt_dict = helixsurf.load_checkpoint(surf, str(conf.load_ckpt))

    # set render options
    opt = helixsurf.RenderOptions()
    helixsurf.setup_render_opts(opt, conf)
    print("Render options:\n", opt)

    # ''' Run : sliding -> clustering -> consistent'''
    # generate_normal_plane(
    #     surf,
    #     conf,
    #     dataset,
    #     mvs_weight=0.1,
    #     plane_slidewindow_size=31,
    #     clean_mesh=True,
    #     clean_percent=0.1,
    #     save_mesh="./testmesh_vis.ply",
    #     constant=True,
    #     vis=True,
    #     vis_dir="tempvis_type1"
    # )

    import time

    time_tic = time.time()
    ray_casting_depth_normal(
        surf,
        conf,
        dataset,
        save_dir="temp_depthnormal",
        # save_dir="temp_depthnormal",
        save_mesh="./testmesh_vis.ply",
        vis=False,
        clean_mesh=True,
        clean_percent=0.05,
        vis_dir="tempvisraycastdepthnormal",
    )
    time_toc = time.time() - time_tic
    print("Time Used: ", time_toc)

