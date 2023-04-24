# Copyright (c) Gorilla-Lab. All rights reserved.
from typing import Tuple

import torch

import helixsurf.libvolume as _C


class _ScatterSum2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(offsets)
        outputs = _C.scatter_sum2d_forward(inputs, offsets)

        return outputs

    @staticmethod
    def backward(ctx, outputs_grad: torch.Tensor) -> Tuple:
        (offsets,) = ctx.saved_tensors
        inputs_grad = _C.scatter_sum2d_backward(outputs_grad, offsets)

        return inputs_grad, None


scatter_sum_2d = _ScatterSum2D.apply


class _ScatterSumBroadcast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(offsets)
        outputs = _C.scatter_sum_broadcast(inputs, offsets)

        return outputs

    @staticmethod
    def backward(ctx, outputs_grad: torch.Tensor) -> Tuple:
        (offsets,) = ctx.saved_tensors
        inputs_grad = _C.scatter_sum_broadcast(outputs_grad, offsets)

        return inputs_grad, None


scatter_sum_broadcast = _ScatterSumBroadcast.apply


# modify the broadcast and scatter_sum from pytorch_scatter https://github.com/rusty1s/pytorch_scatter/blob/master/torch_scatter/scatter.py
def broadcast(offsets: torch.Tensor, inputs: torch.Tensor, dim: int) -> torch.Tensor:
    index = _C.offsets_to_index(offsets)
    if dim < 0:
        dim = inputs.dim() + dim
    for _ in range(0, dim):
        index = index.unsqueeze(0)
    for _ in range(index.dim(), inputs.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(inputs.size())
    return index


def scatter_sum(inputs: torch.Tensor, offsets: torch.Tensor, dim: int = -1) -> torch.Tensor:
    index = broadcast(offsets, inputs, dim).long()
    size = list(inputs.size())
    if index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=inputs.dtype, device=inputs.device)
    return out.scatter_add_(dim, index, inputs)


class _ScatterCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(offsets)
        outputs = _C.scatter_cumsum_forward(inputs, offsets)

        return outputs

    @staticmethod
    def backward(ctx, outputs_grad: torch.Tensor) -> Tuple:
        (offsets,) = ctx.saved_tensors
        inputs_grad = _C.scatter_cumsum_backward(outputs_grad, offsets)

        return inputs_grad, None


scatter_cumsum = _ScatterCumsum.apply


class _ScatterCumprod(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, inputs: torch.Tensor, offsets: torch.Tensor, one_start: bool = False
    ) -> torch.Tensor:
        outputs = _C.scatter_cumprod_forward(inputs, offsets, one_start)
        ctx.save_for_backward(inputs, outputs, offsets)
        ctx.one_start = one_start

        return outputs

    @staticmethod
    def backward(ctx, outputs_grad: torch.Tensor) -> Tuple:
        inputs, outputs, offsets = ctx.saved_tensors
        inputs_grad = _C.scatter_cumprod_backward(
            outputs_grad, inputs, outputs, offsets, ctx.one_start
        )

        return inputs_grad, None, None


scatter_cumprod = _ScatterCumprod.apply
