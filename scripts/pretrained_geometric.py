# adapted from https://github.com/EPFL-VILAB/omnidata
import torch
import torch.nn.functional as F
from torchvision import transforms

import cv2
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import argparse
import os.path
from pathlib import Path
import glob
import sys
import shutil

parser = argparse.ArgumentParser(description="Visualize output for depth or surface normals")

parser.add_argument("--omnidata_path", dest="omnidata_path", help="path to omnidata model")
parser.set_defaults(omnidata_path="/home/yuzh/Projects/omnidata/omnidata_tools/torch/")

parser.add_argument(
    "--pretrained_models", dest="pretrained_models", help="path to pretrained models"
)
parser.set_defaults(
    pretrained_models="/home/yuzh/Projects/omnidata/omnidata_tools/torch/pretrained_models/"
)

parser.add_argument("--task", dest="task", help="normal or depth")
parser.set_defaults(task="NONE")

parser.add_argument("--img_path", dest="img_path", help="path to rgb image")
parser.add_argument("--height", dest="height", default=480, type=int, help="height of image")
parser.add_argument("--width", dest="width", default=640, type=int, help="width of image")
parser.set_defaults(im_name="NONE")

parser.add_argument(
    "--output_path", dest="output_path", help="path to where output image should be stored"
)
parser.set_defaults(store_name="NONE")

args = parser.parse_args()

root_dir = args.pretrained_models
omnidata_path = args.omnidata_path

sys.path.append(args.omnidata_path)
print(sys.path)
from modules.midas.dpt_depth import DPTDepthModel
from data.transforms import get_transform

trans_topil = transforms.ToPILImage()
os.makedirs(args.output_path, exist_ok=True)
os.makedirs(os.path.join(args.output_path, "vis"), exist_ok=True)
# os.system(f"mkdir -p {args.output_path}")
map_location = (
    (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device("cpu")
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

height, width = args.height, args.width
target_size = max(height, width)
left, right = (target_size - width) // 2, (target_size - width) // 2
top, bottom = (target_size - height) // 2, (target_size - height) // 2

image_size: int
# get target task and model
if args.task == "normal":
    image_size = 384

    pretrained_weights_path = root_dir + "omnidata_dpt_normal_v2.ckpt"
    model = DPTDepthModel(backbone="vitb_rn50_384", num_channels=3)  # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if "state_dict" in checkpoint:
        state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    trans_totensor = transforms.Compose(
        [
            transforms.Pad((left, top, right, bottom), fill=0),
            transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
            transforms.CenterCrop(image_size),
            get_transform("rgb", image_size=None),
        ]
    )

elif args.task == "depth":
    image_size = 384
    pretrained_weights_path = (
        root_dir + "omnidata_dpt_depth_v2.ckpt"
    )  # 'omnidata_dpt_depth_v1.ckpt'
    model = DPTDepthModel(backbone="vitb_rn50_384")  # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if "state_dict" in checkpoint:
        state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    trans_totensor = transforms.Compose(
        [
            transforms.Pad((left, top, right, bottom), fill=0),
            transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ]
    )

else:
    print("task should be one of the following: normal, depth")
    sys.exit()

trans_rgb = transforms.Compose(
    [
        transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
        transforms.CenterCrop(image_size),
    ]
)


def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    sorted_img = torch.sort(torch.flatten(img))[0]
    # Remove nan, nan at the end of sort
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    # Remove outliers
    trunc_img = sorted_img[
        int(trunc_value * len(sorted_img)) : int((1 - trunc_value) * len(sorted_img))
    ]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6
    # Replace nan by mean
    img = torch.nan_to_num(img, nan=trunc_mean)
    # Standardize
    img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img


def save_outputs(img_path, output_file_name):
    with torch.no_grad():
        save_path = os.path.join(args.output_path, "vis", f"{output_file_name}_{args.task}.png")

        shutil.copy2(img_path, save_path.replace(f"_{args.task}.png", "_origin.png"))
        # print(f"Reading input {img_path} ...")
        img = Image.open(img_path)

        img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)

        rgb_path = os.path.join(args.output_path, "vis", f"{output_file_name}_rgb.png")
        trans_rgb(img).save(rgb_path)

        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat_interleave(3, 1)

        output = model(img_tensor).clamp(min=0, max=1)

        if args.task == "depth":
            output = output.clamp(0, 1)

            save_array = output.detach().cpu().numpy()[0]
            save_array = cv2.resize(save_array, dsize=(target_size, target_size), interpolation=cv2.INTER_LINEAR)
            save_array = save_array[top: target_size - bottom, left: target_size - right]
            save_array = save_array * 50 + 0.5
            np.save(
                os.path.join(args.output_path, f"{output_file_name}_{args.task}.npy"), save_array
            )
            plt.imsave(save_path, save_array, cmap="viridis")

        else:
            save_array = output.detach().cpu().numpy()[0]
            save_array = np.transpose(save_array, (1, 2, 0))
            save_array = cv2.resize(save_array, dsize=(target_size, target_size), interpolation=cv2.INTER_LINEAR)
            save_array = save_array[top: target_size - bottom, left: target_size - right]
            save_array = save_array * 2 - 1.0
            np.save(
                os.path.join(args.output_path, f"{output_file_name}_{args.task}.npy"), save_array
            )
            normal_map = ((save_array + 1) * 127.5).astype(np.uint8)
            plt.imsave(save_path, normal_map, cmap="viridis")


img_path = Path(args.img_path)
if img_path.is_file():
    save_outputs(args.img_path, os.path.splitext(os.path.basename(args.img_path))[0])
elif img_path.is_dir():
    for f in tqdm(glob.glob(args.img_path + "/*"), desc="Processing"):
        save_outputs(f, os.path.splitext(os.path.basename(f))[0])
else:
    print("invalid file path!")
    sys.exit()

