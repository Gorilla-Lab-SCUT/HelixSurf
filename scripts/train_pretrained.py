# Copyright (c) Gorilla-Lab. All rights reserved.
import argparse
import math
import os

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import helixsurf

try:
    from scripts.generate_normal import ray_casting_depth_normal
except:
    from generate_normal import ray_casting_depth_normal

helixsurf.set_random_seed(123456)
os.environ["PYOPENGL_PLATFORM"] = "egl"


def get_args() -> argparse.Namespace:
    parser = helixsurf.get_default_args()

    parser.add_argument(
        "--mvs_dir",
        type=str,
        help="directory to the mvs",
    )
    parser.add_argument(
        "--image_patch_size",
        "-im_psize",
        type=int,
        default=1,
        help="patch size for an image patch",
    )
    parser.add_argument(
        "--casting",
        action="store_true",
        default=False,
        help="generate depth and normal map using ray casting",
    )
    parser.add_argument(
        "--consistant",
        action="store_true",
        default=False,
        help="confirm the multi view consistency in generated normal",
    )

    # training
    group = parser.add_argument_group("training")
    group.add_argument(
        "--planar_iters",
        type=int,
        default=2500,
        help="iters for planar normal update",
    )
    group.add_argument(
        "--n_epochs",
        type=int,
        default=1,
        help="total number of iters to optimize for learning rate scheduler",
    )
    group.add_argument(
        "--n_iters",
        type=int,
        default=12000,
        help="total number of iters to optimize for learning rate scheduler",
    )
    group.add_argument("--load_ckpt", type=str, default=None, help="the path to load checkpoint")
    group.add_argument("--batch_size", type=int, default=5000, help="batch size")
    group.add_argument("--lr", type=float, default=1e-3, help="learning rate or mlp")

    group.add_argument(
        "--n_train",
        type=int,
        default=None,
        help="Number of training images. Defaults to use all avaiable.",
    )
    group.add_argument(
        "--up_sample_steps",
        type=int,
        default=0,
        help="Number of up sampling. Defaults to 0",
    )

    # visualization
    group = parser.add_argument_group("visualization")
    group.add_argument(
        "--export_mesh",
        type=str,
        default="vis_mc_uniform/sdf_mc.ply",
        help="the path to store the exported mesh",
    )
    group.add_argument(
        "--vis_normal", action="store_true", default=False, help="visualize normal maps"
    )
    group.add_argument(
        "--log_image", action="store_true", default=False, help="logging rendered RGB image"
    )
    group.add_argument(
        "--log_depth", action="store_true", default=False, help="logging rendered depth image"
    )
    group.add_argument("--print_every", type=int, default=10, help="print every")
    group.add_argument("--save_every", type=int, default=1, help="save every x epochs")

    # Patch warping match
    group = parser.add_argument_group("supervision parameters")
    group.add_argument(
        "--loss_type",
        type=str,
        default="smoothl1",
        choices=["smoothl1", "l2", "l1"],
        help="loss type",
    )
    group.add_argument("--ek_lambda", type=float, default=0.05, help="weight for eikonal loss")
    group.add_argument(
        "--ek_bound", type=float, default=1.8, help="the bound of sampling space for eikonal loss"
    )
    group.add_argument(
        "--seed_num",
        type=int,
        default=6,
        help="number of seed for manhattan regularization",
    )
    group.add_argument(
        "--mvs_depth_weight",
        type=float,
        default=1.0,
        help="weight for mvs depth supervision",
    )
    group.add_argument(
        "--mvs_norm_weight",
        type=float,
        default=0.1,
        help="weight for mvs normal supervision",
    )
    group.add_argument(
        "--normal_cos_score_thresh",
        type=float,
        default=0.8,
        help="cos similarity thresh for normal supervision",
    )
    group.add_argument(
        "--manhattan_angle_thresh",
        "-manhattan_thresh",
        type=float,
        default=0.9,
        help="manhattan random direction seed filter thresh.",
    )
    group.add_argument(
        "--plane_normal_weight",
        type=float,
        default=0.01,
        help="weight for mvs plane normal supervision",
    )
    group.add_argument(
        "--plane_manhattan_normal_weight",
        type=float,
        default=0.03,
        help="weight for mvs plane manhattan normal supervision",
    )

    # ray casting marching cube
    group = parser.add_argument_group("ray casting marching cubes parameters")
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


def train_epoch(
    surf: helixsurf.HelixSurf,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    opt: helixsurf.RenderOptions,
    conf: omegaconf.OmegaConf,
    dataset_train: helixsurf.ScanNetPretrainedDataset,
    gstep_id_base: int,
    epoch_id: int,
    patch_size: int = 1,
) -> int:
    surf.update_occupancy_grid(opt=opt)
    surf.export_mesh(filename=conf.export_mesh)

    print("Train epoch:")
    num_rays = dataset_train.rays.origins.size(0)
    num_iters_per_epoch = (num_rays - 1) // conf.batch_size + 1
    pbar = tqdm(enumerate(range(0, num_rays, conf.batch_size)), total=num_iters_per_epoch)
    stats = {"mse": 0.0, "psnr": 0.0, "invsqr_mse": 0.0}

    rgb_criternion: nn.Module
    if conf.loss_type == "smoothl1":
        rgb_criternion = nn.SmoothL1Loss(beta=0.1)
    elif conf.loss_type == "l2":
        rgb_criternion = nn.MSELoss()
    elif conf.loss_type == "l1":  # does not work
        rgb_criternion = nn.L1Loss()
    depth_criternion = nn.L1Loss(reduction="none")
    cos_sim = nn.CosineSimilarity()

    surf.export_mesh(filename=conf.export_mesh)

    for iter_id, batch_begin in pbar:
        gstep_id = iter_id + gstep_id_base

        update_grid_freq = 16
        if (gstep_id + 1) % update_grid_freq == 0:
            surf.update_occupancy_grid(opt=opt)

        # get the elements for one batch
        batch_end = min(batch_begin + conf.batch_size, num_rays)
        batch_origins = dataset_train.rays.origins[batch_begin:batch_end]
        batch_dirs = dataset_train.rays.dirs[batch_begin:batch_end]
        rgb_gt = dataset_train.rays.gt[batch_begin:batch_end]
        depth_gt = dataset_train.rays.depth_gt[batch_begin:batch_end]
        if dataset_train.input_normal:
            normal_gt = dataset_train.rays.normal_gt[batch_begin:batch_end]
        batch_pretrained_normal = dataset_train.rays.pretrained_normal[batch_begin:batch_end]
        batch_pretrained_mask = dataset_train.rays.pretrained_mask[batch_begin:batch_end]
        rays = helixsurf.Rays(batch_origins, batch_dirs)

        # export mesh per 1000 iterations
        if (gstep_id + 1) % 1000 == 0:
            surf.export_mesh(
                filename=str(conf.export_mesh).replace(".ply", f"_iter_{gstep_id}.ply"),
                resolution=256 if epoch_id < 90 else 512,
                batch_size=64**3,
            )

        cos_anneal = min(1.0, (gstep_id + 1) / (25000 * 512 / conf.batch_size))
        optimizer.zero_grad()

        up_sample_steps = conf.up_sample_steps if gstep_id >= 10000 else 0
        (rgb_pred, depth_pred, normal_pred, normal_eik, sval, _,) = surf.train_render(
            rays,
            opt,
            cos_anneal_ratio=cos_anneal,
            render_normal=True,
            ek_bound=conf.ek_bound,
            up_sample_steps=up_sample_steps,
        )

        depth_mask = depth_gt > 0
        normals_normalized = F.normalize(normal_pred, p=2, dim=-1).clamp(-1.0, 1.0)

        ######################################
        """ RGB loss """
        rgb_loss = rgb_criternion(rgb_pred, rgb_gt)
        rgb_confidence = 1.0 - depth_criternion(
            rgb_pred[depth_mask.squeeze()], rgb_gt[depth_mask.squeeze()]
        ).mean(-1)
        """ Eikonal loss """
        ek_loss = float(conf.ek_lambda) * ((normal_eik - 1.0) ** 2).mean()
        loss = rgb_loss + ek_loss

        """ Depth loss """
        mvs_depth_loss = 0.0
        mvs_depth_weight = conf.mvs_depth_weight
        # weight decay for depth
        if gstep_id > 5000:
            mvs_depth_weight *= 0.5
        if gstep_id > 7500:
            mvs_depth_weight *= 0.5

        mvs_depth_loss = rgb_confidence * depth_criternion(
            depth_pred[depth_mask], depth_gt[depth_mask]
        )
        if gstep_id > 5000:
            mvs_depth_loss = mvs_depth_loss.clamp(max=0.5)  # Clamp too large depth deviation

        mvs_depth_loss = mvs_depth_loss.mean()
        loss += mvs_depth_weight * mvs_depth_loss

        """ MVS Normal loss """
        mvs_norm_loss = 0.0
        mvs_norm_weight = conf.mvs_norm_weight
        # weight decay for normal
        if gstep_id > 5000:
            mvs_norm_weight = 0.05

        if dataset_train.input_normal and True:
            masked_normals = normals_normalized[depth_mask.squeeze()]
            masked_normal_gt = normal_gt[depth_mask.squeeze()]

            cos_score_mvs = cos_sim(masked_normals, masked_normal_gt)
            mvs_normal_mask = cos_score_mvs > min(
                conf.normal_cos_score_thresh, (gstep_id / conf.n_iters)
            )
            # l1 loss
            mvs_norm_loss = (
                mvs_norm_weight
                * (
                    rgb_confidence[mvs_normal_mask, None]
                    * (masked_normals[mvs_normal_mask] - masked_normal_gt[mvs_normal_mask])
                )
                .abs()
                .mean()
            )
            loss += mvs_norm_loss

        """ Pretrained loss"""
        textureless_normal_loss = 0.0
        textureless_normal_weight = 0.1
    
        selected_textureless_normals = normals_normalized
        selected_textureless_normals_gt = batch_pretrained_normal
        selected_textureless_normals = normals_normalized[
            batch_pretrained_mask
        ]
        selected_textureless_normals_gt = batch_pretrained_normal[
            batch_pretrained_mask
        ]
        textureless_normal_l1 = (
            torch.abs(selected_textureless_normals - selected_textureless_normals_gt)
            .sum(dim=-1)
            .mean()
        )
        textureless_normal_cos = (
            1.0 - cos_sim(selected_textureless_normals, selected_textureless_normals_gt)
        ).mean()
        textureless_normal_loss = textureless_normal_l1 + textureless_normal_cos
        loss += textureless_normal_weight * textureless_normal_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        rgb_pred.clamp_max_(1.0)
        mse = F.mse_loss(rgb_gt, rgb_pred)

        # Stats
        mse_num: float = mse.detach().item()
        psnr = -10.0 * math.log10(mse_num)
        stats["mse"] += mse_num
        stats["psnr"] += psnr
        stats["invsqr_mse"] += 1.0 / mse_num**2

        if (iter_id + 1) % conf.print_every == 0:
            # Print averaged stats
            pbar.set_description(
                f"epoch {epoch_id} iter {gstep_id} psnr={psnr:.2f} ek={ek_loss:.4f} "
                f"mvs_dep={mvs_depth_loss:.4f} mvs_nor={mvs_norm_loss:.4f} "
                f"textureless={textureless_normal_loss:.4f}"
            )
            # Logging in tensorboard
            for stat_name, stat_val in stats.items():
                stat_val = stat_val / conf.print_every
                summary_writer.add_scalar(stat_name, stat_val, global_step=gstep_id)
                stat_val = 0.0
            summary_writer.add_scalar(
                "Params/learning_rate",
                optimizer.param_groups[0]["lr"],
                global_step=gstep_id,
            )
            summary_writer.add_scalar("Params/cos_anneal_ratio", cos_anneal, global_step=gstep_id)
            summary_writer.add_scalar("Params/s_val", sval, global_step=gstep_id)
            summary_writer.add_scalar("Loss/total", loss, global_step=gstep_id)
            summary_writer.add_scalar("Loss/rgb", rgb_loss, global_step=gstep_id)
            summary_writer.add_scalar("Loss/depth", mvs_depth_loss, global_step=gstep_id)
            summary_writer.add_scalar("Loss/normal", mvs_norm_loss, global_step=gstep_id)
            summary_writer.add_scalar("Loss/ek", ek_loss, global_step=gstep_id)

    gstep_id_base += len(pbar)

    return gstep_id_base


if __name__ == "__main__":
    # prase args
    args = get_args()
    conf = helixsurf.merge_config_file(args)
    print("Config:\n", omegaconf.OmegaConf.to_yaml(conf))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # init train directory and tensorboard
    os.makedirs(conf.train_dir, exist_ok=True)
    summary_writer = SummaryWriter(conf.train_dir)

    # backup
    helixsurf.backup(
        backup_dir = os.path.join(conf.train_dir, "backup"),
        backup_list = ["helixsurf", conf.config, "cpp", __file__],
        contain_suffix = ["*.py", "*.hpp", "*.cuh", "*.cpp", "*.cu"],
    )

    # initialize model
    conf.model.bound = conf.bound
    conf.model.grid_res = conf.grid_res
    print(f"Using Occupancy Grids (bound : {conf.bound} | res : {conf.grid_res})")
    surf = helixsurf.HelixSurf(**conf.model).to(device)
    surf.train()
    optimizer = optim.Adam(surf.parameters(), lr=conf.lr)
    scheduler = helixsurf.NeusScheduler(
        optimizer, warm_up_end=800, total_steps=conf.n_iters * conf.n_epochs
    )

    # set render options
    opt = helixsurf.RenderOptions()
    helixsurf.setup_render_opts(opt, conf)
    print("Render options:\n", opt)

    # init dataset
    factor = 1
    img_patch_size = conf.image_patch_size
    train_dataset = helixsurf.datasets[conf.dataset_type](
        conf.data_dir,
        mvs_root=conf.mvs_dir,
        scene=conf.scene,
        split="train",
        device=device,
        factor=factor,
        patch_size=img_patch_size,
        n_images=conf.n_train,
    )

    # prepare for training
    ckpt_path = os.path.join(conf.train_dir, "ckpt.pth")
    batch_size = conf.batch_size

    num_epochs = args.n_epochs
    gstep_id_base = 0
    print(f"Total epoches {num_epochs}")

    for epoch_id in range(num_epochs):
        # wheter to reload from a checkpoint
        if conf.load_ckpt is not None:
            ckpt_dict = helixsurf.load_checkpoint(surf, str(conf.load_ckpt))
            print(f"loadckpt from {conf.load_ckpt}")
            epoch_id = int(ckpt_dict["meta"]["epoch_id"])
            gstep_id_base = int(ckpt_dict["meta"]["gstep_id_base"])

        # whether upsampling (update the grid sample res)
        upsampling = (epoch_id + 1) % conf.ups_epoch == 0 and epoch_id > 0
        if upsampling:
            opt.grid_sample_res = min(opt.grid_sample_res * 2, opt.max_grid_sample_res)
            print(f"---------- Begin Upsampling ---------")

        # training
        if img_patch_size > 1:
            train_dataset.shuffle_rays_withuv(
                patch_size=img_patch_size, epoch_size=conf.batch_size * conf.n_iters
            )
        else:
            train_dataset.shuffle_rays(epoch_size=conf.batch_size * conf.n_iters)

        gstep_id_base = train_epoch(
            surf,
            optimizer,
            scheduler,
            opt,
            conf,
            train_dataset,
            gstep_id_base,
            epoch_id,
            img_patch_size,
        )

        # save
        if conf.save_every > 0 and (epoch_id + 1) % max(factor, conf.save_every) == 0:
            print("Saving", ckpt_path)
            helixsurf.save_checkpoint(
                surf,
                ckpt_path,
                optimizer,
                scheduler,
                dict(epoch_id=epoch_id + 1, gstep_id_base=gstep_id_base + 1),
            )
            helixsurf.save_checkpoint(
                surf,
                ckpt_path.replace(".pth", f"_{epoch_id + 1}.pth"),
                optimizer,
                scheduler,
                dict(epoch_id=epoch_id + 1, gstep_id_base=gstep_id_base + 1),
            )

        # ray casting depth and normals
        if epoch_id == 0 and conf.casting:
            ray_casting_depth_normal(
                surf,
                conf,
                train_dataset,
                save_dir=os.path.join(args.mvs_dir, f"ep_{epoch_id + 1}"),
                save_mesh=os.path.join(args.mvs_dir, f"ep_{epoch_id + 1}.ply"),
                disturb_depth_noise=0.3,
                disturb_normal_noise=0.3,
                clean_mesh=True,
                clean_percent=0.05,  # not too large
                vis=conf.vis_normal,
                vis_dir=os.path.join(args.mvs_dir, f"ep_{epoch_id + 1}_vis"),
            )
