# Copyright (c) Gorilla-Lab. All rights reserved.
import warnings
from typing import Any, Callable, List

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def get_expon_lr_func(
    lr_init: float,
    lr_final: float,
    lr_delay_steps: int = 0,
    lr_delay_mult: float = 1.0,
    max_steps: int = 1000000,
) -> Callable:
    """
    Continuous learning rate decay function. Adapted from JaxNeRF

    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.

    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


class NeusScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        learning_rate_alpha=0.05,
        warm_up_end: int = 5000,
        total_steps: int = 300000,
        min_lr: int = 1e-10,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        # Validate optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.warm_up_end = warm_up_end
        self.learning_rate_alpha = learning_rate_alpha

        # Validate total_steps
        if total_steps is not None:
            self.total_steps = total_steps
        else:
            raise ValueError("You must define total_steps")

        super(NeusScheduler, self).__init__(optimizer, last_epoch, verbose)

    def _format_param(self, name: str, optimizer: Optimizer, param: Any) -> List[torch.Tensor]:
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError(
                    "expected {} values for {}, got {}".format(
                        len(optimizer.param_groups), name, len(param)
                    )
                )
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        step_num = self.last_epoch

        if step_num > self.total_steps:
            raise ValueError(
                "Tried to step {} times. The specified number of total steps is {}".format(
                    step_num + 1, self.total_steps
                )
            )

        if step_num < self.warm_up_end:
            coeff = step_num / self.warm_up_end
        else:
            progress = (step_num - self.warm_up_end) / (self.total_steps - self.warm_up_end)
            # progress = min(progress, 1 - 1e-2)
            coeff = (np.cos(np.pi * progress) + 1.0) * 0.5 * (
                1 - self.learning_rate_alpha
            ) + self.learning_rate_alpha

        return [(base_lr - self.min_lr) * coeff + self.min_lr for base_lr in self.base_lrs]


class ExponentialLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, decay_epochs, gamma=0.1, last_epoch=-1):
        self.decay_epochs = decay_epochs
        self.gamma = gamma
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr * self.gamma ** (self.last_epoch / self.decay_epochs)
            for base_lr in self.base_lrs
        ]
