__all__ = [
    "linear_backsub_gather",
    "linear_backsub_gather_lower",
    "linear_backsub_slice",
    "linear_backsub_slice_lower",
]

import torch
from torch import Tensor

from ...utils import gen_slice_list


def linear_backsub_gather(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    ori_shape: tuple[int, ...],
    dim: int,
    index: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    dtype = lower_symbnd_coeffs.dtype
    device = lower_symbnd_coeffs.device
    kwargs = {"dtype": dtype, "device": device}
    n_b = lower_symbnd_coeffs.shape[0]
    shape = (n_b, *ori_shape)
    new_lower_symbnd_coeffs_tensor = torch.zeros(shape, **kwargs)
    new_upper_symbnd_coeffs_tensor = new_lower_symbnd_coeffs_tensor.clone()
    dim = dim + 1
    index = index.unsqueeze(0).expand(n_b, -1)

    new_lower_symbnd_coeffs_tensor.scatter_(dim, index, lower_symbnd_coeffs)
    new_upper_symbnd_coeffs_tensor.scatter_(dim, index, upper_symbnd_coeffs)

    new_lower_symbnd_const = lower_symbnd_const
    new_upper_symbnd_const = upper_symbnd_const

    return (
        new_lower_symbnd_coeffs_tensor,
        new_lower_symbnd_const,
        new_upper_symbnd_coeffs_tensor,
        new_upper_symbnd_const,
    )


def linear_backsub_gather_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    ori_shape: tuple[int, ...],
    dim: int,
    index: Tensor,
) -> tuple[Tensor, Tensor]:
    dtype = lower_symbnd_coeffs.dtype
    device = lower_symbnd_coeffs.device
    kwargs = {"dtype": dtype, "device": device}
    n_b = lower_symbnd_coeffs.shape[0]
    shape = (n_b, *ori_shape)
    new_lower_symbnd_coeffs_tensor = torch.zeros(shape, **kwargs)
    dim = dim + 1
    index = index.unsqueeze(0).expand(n_b, -1)

    new_lower_symbnd_coeffs_tensor.scatter_(dim, index, lower_symbnd_coeffs)

    new_lower_symbnd_const = lower_symbnd_const

    return new_lower_symbnd_coeffs_tensor, new_lower_symbnd_const


def linear_backsub_slice(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    ori_shape: tuple[int, ...],
    starts: list[int],
    ends: list[int],
    axes: list[int],
    steps: list[int],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    dtype = lower_symbnd_coeffs.dtype
    device = lower_symbnd_coeffs.device
    kwargs = {"dtype": dtype, "device": device}

    n_b = lower_symbnd_coeffs.shape[0]
    shape = (n_b, *ori_shape)

    new_lower_symbnd_coeffs_tensor = torch.zeros(shape, **kwargs)
    new_upper_symbnd_coeffs_tensor = new_lower_symbnd_coeffs_tensor.clone()

    slicing = gen_slice_list(len(ori_shape), starts, ends, axes, steps)

    new_lower_symbnd_coeffs_tensor[:, *slicing] = lower_symbnd_coeffs
    new_upper_symbnd_coeffs_tensor[:, *slicing] = upper_symbnd_coeffs

    new_lower_symbnd_const = lower_symbnd_const
    new_upper_symbnd_const = upper_symbnd_const

    return (
        new_lower_symbnd_coeffs_tensor,
        new_lower_symbnd_const,
        new_upper_symbnd_coeffs_tensor,
        new_upper_symbnd_const,
    )


def linear_backsub_slice_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    ori_shape: tuple[int, ...],
    starts: list[int],
    ends: list[int],
    axes: list[int],
    steps: list[int],
) -> tuple[Tensor, Tensor]:
    dtype = lower_symbnd_coeffs.dtype
    device = lower_symbnd_coeffs.device
    kwargs = {"dtype": dtype, "device": device}

    n_b = lower_symbnd_coeffs.shape[0]
    shape = (n_b, *ori_shape)

    new_lower_symbnd_coeffs_tensor = torch.zeros(shape, **kwargs)

    slicing = gen_slice_list(len(ori_shape), starts, ends, axes, steps)

    new_lower_symbnd_coeffs_tensor[:, *slicing] = lower_symbnd_coeffs

    new_lower_symbnd_const = lower_symbnd_const

    return new_lower_symbnd_coeffs_tensor, new_lower_symbnd_const
