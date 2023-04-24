# Copyright (c) Gorilla-Lab. All rights reserved.

from .dataset_base import DatasetBase
from .scannet import ScanNetDataset
from .scannet_pretrained import ScanNetPretrainedDataset


def scannet_dataset(root: str, *args, **kwargs) -> DatasetBase:
    print("Defaulting to extended ScanNet dataset")
    return ScanNetDataset(root, *args, **kwargs)

def scannet_pretrained_dataset(root: str, *args, **kwargs) -> DatasetBase:
    print("Defaulting to extended ScanNet dataset")
    return ScanNetPretrainedDataset(root, *args, **kwargs)

datasets = {
    "scannet": scannet_dataset,
    "scannet_pretrained": scannet_pretrained_dataset,
}

__all__ = ["datasets", "DatasetBase", "ScanNetDataset", "ScanNetPretrainedDataset"]
