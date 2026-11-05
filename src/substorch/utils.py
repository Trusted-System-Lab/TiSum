__all__ = [
    "cal_l_argmax_maxpool2d",
    "unfold_pre_bound_maxpool2d",
    "get_pool_idxs",
    "gen_slice_list",
]

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import Tensor


def cal_l_argmax_maxpool2d(
    unfold_pre_lower_bound: Tensor,
    unfold_pre_upper_bound: Tensor,
) -> tuple[Tensor, Tensor]:
    """Calculate the maximum value and the index of the maximum value of the input tensor."""

    """
    The output of torch.max for the same values is not deterministic. The order Will 
    affect the result of the backward pass.

    (1) Search from the left to the right.
    l_max, l_argmax = l.max(dim=1)

    (2) Search from the right to the left.
    l_max, l_argmax = torch.flip(l, [1]).max(dim=1)
    l_argmax = l.shape[1] - l_argmax - 1

    (3) Set r as the second index when finding the maximum value.
    r = u
    l_max, _ = l.max(dim=1)
    max_mask = l == l_max.unsqueeze(1)
    r_masked = r.where(max_mask, torch.tensor(float("inf")))
    l_argmax = r_masked.argmin(dim=1)

    (4) Set r as the second index when finding the maximum value.
    r_masked = r.where(max_mask, torch.tensor(-float("inf")))
    l_argmax = r_masked.argmax(dim=1)

    """

    r = unfold_pre_upper_bound - unfold_pre_lower_bound
    l_max, _ = unfold_pre_lower_bound.max(dim=1)
    max_mask = unfold_pre_lower_bound == l_max.unsqueeze(1)
    r_masked = r.where(max_mask, torch.tensor(-float("inf")))
    l_argmax = r_masked.argmax(dim=1)

    return l_max, l_argmax


def unfold_pre_bound_maxpool2d(
    pre_lower_bound: Tensor,
    pre_upper_bound: Tensor,
    *,
    kernel_size: tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    dilation: int | tuple[int, int],
) -> tuple[Tensor, Tensor]:
    kwargs = {
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
    }
    num_channels = pre_lower_bound.shape[0]
    num_kernel_eles = kernel_size[0] * kernel_size[1]
    pre_lower_bound = (
        F.unfold(pre_lower_bound, **kwargs)
        .reshape((num_channels, num_kernel_eles, -1))
        .permute(0, 2, 1)
        .reshape((-1, num_kernel_eles))
    )
    pre_upper_bound = (
        F.unfold(pre_upper_bound, **kwargs)
        .reshape((num_channels, num_kernel_eles, -1))
        .permute(0, 2, 1)
        .reshape((-1, num_kernel_eles))
    )

    return pre_lower_bound, pre_upper_bound


@lru_cache(maxsize=None)
def get_pool_idxs(
    input_size: tuple[int, int, int],
    kernel_size: tuple[int, int],
    padding: int,
    dilation: int,
    stride: int,
    num_kernels: int,
    num_kernel_eles: int,
) -> Tensor:
    c, h, w = input_size
    pool_idxs = (
        F.unfold(
            torch.arange(c * h * w).reshape(c, h, w).to(dtype=torch.float32),
            kernel_size,
            dilation,
            padding,
            stride,
        )
        .reshape((c, num_kernel_eles, num_kernels))
        .permute((0, 2, 1))
        .reshape((c * num_kernels, num_kernel_eles))
        .to(dtype=torch.int32)
    )

    return pool_idxs


def gen_slice_list(
    dims: int,
    starts: list[int],
    ends: list[int],
    axes: list[int],
    steps: list[int],
) -> list[slice]:
    slicing = []
    for axis in range(dims):
        if axis in axes:
            idx = axes.index(axis)
            start = starts[idx]
            end = ends[idx]
            step = steps[idx]
            slicing.append(slice(start, end, step))
        else:
            slicing.append(slice(None))

    return slicing
