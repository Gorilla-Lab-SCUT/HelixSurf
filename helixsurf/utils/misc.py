# Copyright (c) Gorilla-Lab. All rights reserved.
import cv2
from typing import Optional, Tuple
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.vq import kmeans as sp_kmeans
from scipy.spatial import cKDTree


def compute_ssim(
    img0: torch.Tensor,
    img1: torch.Tensor,
    max_val: float = 1.0,
    filter_size: float = 11,
    filter_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    return_map: bool = False,
) -> torch.Tensor:
    """Computes SSIM from two images.
    This function was modeled after tf.image.ssim, and should produce comparable output.

    Args:
        img0 (torch.Tensor): An image of size [..., width, height, num_channels].
        img1 (torch.Tensor): An image of size [..., width, height, num_channels].
        max_val (float, optional): The maximum magnitude that `img0` or `img1` can have. Defaults to 1.0.
        filter_size (float, optional): Window size. Defaults to 11.
        filter_sigma (float, optional): The bandwidth of the Gaussian used for filtering. Defaults to 1.5.
        k1 (float, optional): One of the SSIM dampening parameters. Defaults to 0.01.
        k2 (float, optional): One of the SSIM dampening parameters. Defaults to 0.03.
        return_map (bool, optional): If True, will cause the per-pixel SSIM "map" to returned. Defaults to False.

    Returns:
        torch.Tensor: Each image's mean SSIM, or a tensor of individual values if `return_map`.
    """

    device = img0.device
    ori_shape = img0.size()
    width, height, num_channels = ori_shape[-3:]
    img0 = img0.view(-1, width, height, num_channels).permute(0, 3, 1, 2)
    img1 = img1.view(-1, width, height, num_channels).permute(0, 3, 1, 2)

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((torch.arange(filter_size, device=device) - hw + shift) / filter_sigma) ** 2
    filt = torch.exp(-0.5 * f_i)
    filt /= torch.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    # z is a tensor of size [B, H, W, C]
    def filt_fn1(z):
        return F.conv2d(
            z,
            filt.view(1, 1, -1, 1).repeat(num_channels, 1, 1, 1),
            padding=[hw, 0],
            groups=num_channels,
        )

    def filt_fn2(z):
        return F.conv2d(
            z,
            filt.view(1, 1, 1, -1).repeat(num_channels, 1, 1, 1),
            padding=[0, hw],
            groups=num_channels,
        )

    # Vmap the blurs to the tensor size, and then compose them.
    def filt_fn(z):
        return filt_fn1(filt_fn2(z))

    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = torch.clamp(sigma00, min=0.0)
    sigma11 = torch.clamp(sigma11, min=0.0)
    sigma01 = torch.sign(sigma01) * torch.min(torch.sqrt(sigma00 * sigma11), torch.abs(sigma01))

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = torch.mean(ssim_map.reshape([-1, num_channels * width * height]), dim=-1)
    return ssim_map if return_map else ssim


def viridis_cmap(gray: np.ndarray) -> np.ndarray:
    """
    Visualize a single-channel image using matplotlib's viridis color map
    yellow is high value, blue is low
    :param gray: np.ndarray, (H, W) or (H, W, 1) unscaled
    :return: (H, W, 3) float32 in [0, 1]
    """
    colored = plt.cm.viridis(plt.Normalize()(gray.squeeze()))[..., :-1]
    return colored.astype(np.float32)


def save_img(img: np.ndarray, path: str) -> None:
    """Save an image to disk. Image should have values in [0,1]."""
    img = np.array((np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def initialize(X: torch.Tensor, k: int, seed: Optional[int]) -> torch.Tensor:
    """initialize cluster centers
    Args:
        X (torch.Tensor): matrix
        k (int): number of clusters
        seed (Optional[int]): seed for kmeans
    Returns:
        torch.Tensor: initial state
    """
    num_samples = len(X)
    if seed == None:
        indices = np.random.choice(num_samples, k, replace=False)
    else:
        np.random.seed(seed)
        indices = np.random.choice(num_samples, k, replace=False)
    initial_state = X[indices]
    return initial_state


def k_means_fast(
    X: torch.Tensor,
    k: int,
    distance: str = "euclidean",
    tol: float = 1e-5,
    iter_limit=20,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor]:
    """k-means
    Args:
        X (torch.Tensor): matrix
        k (int): number of clusters
        distance (str, optional): distance [options: "euclidean", "cosine"]. Defaults to "euclidean".
        tol (float, optional): threshold. Defaults to 1e-4.
        iter_limit (int, optional): hard limit for max number of iterations. Defaults to 20.
        seed (Optional[int], optional): random seed. Defaults to None.
    Raises:
        NotImplementedError: invalid distance metric
    Returns:
        Tuple[torch.Tensor]: cluster centers & cluster ids
    """
    if distance == 'euclidean':
        pairwise_distance_function = partial(pairwise_distance)
    elif distance == 'cosine':
        pairwise_distance_function = partial(pairwise_cosine)
    else:
        raise NotImplementedError

    # initialize
    centers = initialize(X, k, seed=seed)

    best_shift = torch.inf
    best_state = None
    best_cluster = None
    iteration = 0
    while True:
        dis = pairwise_distance_function(X, centers)

        cluster_ids = torch.argmin(dis, dim=1)

        centers_pre = centers.clone()

        for index in range(k):
            selected = torch.nonzero(cluster_ids == index).squeeze()

            selected = torch.index_select(X, 0, selected)

            # https://github.com/subhadarship/kmeans_pytorch/issues/16
            if selected.shape[0] == 0:
                selected = X[torch.randint(len(X), (1,))]

            centers[index] = selected.mean(dim=0)

        center_shift = torch.sum(torch.sqrt(torch.sum((centers - centers_pre) ** 2, dim=1)))

        # increment iteration
        iteration = iteration + 1

        if center_shift ** 2 < best_shift:
            best_shift = center_shift ** 2
            best_state = centers
            best_cluster = cluster_ids
        if center_shift**2 < tol:
            break
        if iter_limit != 0 and iteration >= iter_limit:
            break

    if best_state is not None:
        return best_state, best_cluster
    else:
        return centers, cluster_ids


def pairwise_distance(data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1: torch.Tensor, data2: torch.Tensor) -> torch.Tensor:
    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis


def k_means(points: torch.Tensor, k: int, **kwargs):
    """
    Find k centroids that attempt to minimize the k- means problem:
    https://en.wikipedia.org/wiki/Metric_k-center
    Parameters
    ----------
    points:  (n, d) float
      Points in space
    k : int
      Number of centroids to compute
    **kwargs : dict
      Passed directly to scipy.cluster.vq.kmeans
    Returns
    ----------
    centroids : (k, d) float
      Points in some space
    labels: (n) int
      Indexes for which points belong to which centroid
    """
    device = points.device
    points = np.asanyarray(points.cpu().numpy(), dtype=np.float32)
    points_std = points.std(axis=0)
    points_std[points_std < 1e-12] = 1
    whitened = points / points_std
    centroids_whitened, _ = sp_kmeans(whitened, k, **kwargs)
    centroids = centroids_whitened * points_std

    # find which centroid each point is closest to
    tree = cKDTree(centroids)
    labels = tree.query(points, k=1)[1]

    return torch.from_numpy(centroids).float().to(device), \
           torch.from_numpy(labels).to(torch.int32).to(device)


def fast_k_means(points: torch.Tensor, k: int, distance: str ='euclidean'):
    """
    Parameters
    ----------
    points:  (n, d) float
      Points in space
    k : int
      Number of centroids to compute
    distance : ['euclidean' | 'cosine'] 
    Returns
    ----------
    centroids : (k, d) float
      Points in some space
    labels: (n) int
      Indexes for which points belong to which centroid
    """
    points_std = points.std(dim=0)
    points_std[points_std < 1e-12] = 1
    whitened = points / points_std

    centroids_whitened, labels = k_means_fast(
      whitened, k, distance=distance
    )
    centroids = centroids_whitened * points_std

    return centroids, labels
