# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import time
from collections import OrderedDict
from typing import Callable, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def is_module_wrapper(module: nn.Module) -> bool:
    """Check if a module is a module wrapper.
    The following modules are regarded as
    module wrappers: DataParallel, DistributedDataParallel
    Args:
        module (nn.Module): The module to be checked.
    Returns:
        bool: True if the input module is a module wrapper.
    """
    module_wrappers = (DataParallel, DistributedDataParallel)
    return isinstance(module, module_wrappers)


def weights_to_cpu(state_dict: Dict) -> Dict:
    r"""Copy a model state_dict to cpu.
    Args:
        state_dict (OrderedDict): Model weights on GPU.
    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


def _save_to_state_dict(module: nn.Module, destination: Dict, prefix: str, keep_vars: bool) -> None:
    r"""Saves module state to `destination` dictionary.
    This method is modified from :meth:`torch.nn.Module._save_to_state_dict`.
    Args:
        module (nn.Module): The module to generate state_dict.
        destination (dict): A dict where state will be stored.
        prefix (str): The prefix for parameters and buffers used in this
            module.
    """
    for name, param in module._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()
    for name, buf in module._buffers.items():
        # remove check of _non_persistent_buffers_set to allow nn.BatchNorm2d
        if buf is not None:
            destination[prefix + name] = buf if keep_vars else buf.detach()


def get_state_dict(
    module: nn.Module,
    destination: Optional[OrderedDict] = None,
    prefix: str = "",
    keep_vars: bool = False,
) -> Dict:
    r"""Returns a dictionary containing a whole state of the module.
    Both parameters and persistent buffers (e.g. running averages) are
    included. Keys are corresponding parameter and buffer names.
    This method is modified from :meth:`torch.nn.Module.state_dict` to
    recursively check parallel module in case that the model has a complicated
    structure, e.g., nn.Module(nn.Module(DDP)).
    Args:
        module (nn.Module): The module to generate state_dict.
        destination (OrderedDict): Returned dict for the state of the
            module.
        prefix (str): Prefix of the key.
        keep_vars (bool): Whether to keep the variable property of the
            parameters. Default: False.
    Returns:
        dict: A dictionary containing a whole state of the module.
    """
    # recursively check parallel module in case that the model has a
    # complicated structure, e.g., nn.Module(nn.Module(DDP))
    if is_module_wrapper(module):
        module = module.module

    # below is the same as torch.nn.Module.state_dict()
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    destination._metadata[prefix[:-1]] = local_metadata = dict(version=module._version)
    _save_to_state_dict(module, destination, prefix, keep_vars)
    for name, child in module._modules.items():
        if child is not None:
            get_state_dict(child, destination, prefix + name + ".", keep_vars=keep_vars)
    for hook in module._state_dict_hooks.values():
        hook_result = hook(module, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination


def save_checkpoint(
    model: nn.Module,
    filename: str,
    optimizer: Optional[Union[Optimizer, Dict]] = None,
    scheduler: Optional[Union[_LRScheduler, Dict]] = None,
    meta: Optional[Dict] = None,
) -> None:
    r"""Save checkpoint to file.
    The checkpoint will have 3 fields:
        ``meta``, ``state_dict`` and ``optimizer``.
        By default ``meta`` will contain version and time info.
    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.
    """
    # prevent save incomplete checkpoint due to key interrupt
    if meta is None:
        meta = {}
    elif not isinstance(meta, Dict):
        raise TypeError(f"meta must be a dict or None, but got {type(meta)}")
    meta.update(time=time.asctime())

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if is_module_wrapper(model):
        model = model.module

    checkpoint = {"meta": meta, "model": weights_to_cpu(get_state_dict(model))}
    # save optimizer state dict in the checkpoint
    if optimizer is not None:
        if isinstance(optimizer, Optimizer):
            checkpoint["optimizer"] = optimizer.state_dict()
        elif isinstance(optimizer, dict):
            checkpoint["optimizer"] = {}
            for name, optim in optimizer.items():
                checkpoint["optimizer"][name] = optim.state_dict()
        else:
            raise TypeError(
                f"Optimizer should be dict or torch.optim.Optimizer but got {type(optimizer)}"
            )

    # save lr_scheduler state dict in the checkpoint
    if scheduler is not None:
        if isinstance(scheduler, _LRScheduler):
            checkpoint["scheduler"] = scheduler.state_dict()
        elif isinstance(scheduler, Dict):
            checkpoint["scheduler"] = {}
            for name, sche in scheduler.items():
                checkpoint["scheduler"][name] = sche.state_dict()
        else:
            raise TypeError(
                f"scheduler should be dict or torch.optim.lr_scheduler._LRScheduler/ReduceLROnPlateau "
                f"but got {type(scheduler)}"
            )

    # immediately flush buffer
    with open(filename, "wb") as f:
        torch.save(checkpoint, f)
        f.flush()


def load_state_dict(
    module: nn.Module,
    state_dict: Dict,
    strict: bool = False,
) -> None:
    """Load state_dict to a module.
    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.
    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module"s
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=""):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if is_module_wrapper(module):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            all_missing_keys,
            unexpected_keys,
            err_msg,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [key for key in all_missing_keys if "num_batches_tracked" not in key]

    if unexpected_keys:
        err_msg.append(f"unexpected key in source state_dict: {', '.join(unexpected_keys)}\n")
    if missing_keys:
        err_msg.append(f"missing keys in source state_dict: {', '.join(missing_keys)}\n")

    if len(err_msg) > 0:
        err_msg.insert(0, "The model and loaded state dict do not match exactly\n")
        err_msg = "\n".join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        print(err_msg)


def load_checkpoint(
    model: nn.Module,
    filename: str,
    map_location: Optional[Union[str, Callable]] = None,
    strict: bool = True,
) -> Dict:
    r"""Load checkpoint from a file or URI.
    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL.
        map_location (func): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = torch.load(filename, map_location=map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")
    # get model state_dict from checkpoint
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    # load state_dict
    load_state_dict(model, state_dict, strict)
    return checkpoint
