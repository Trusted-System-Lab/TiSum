__all__ = [
    "batched_linear_backsub_slice",
    "batched_linear_backsub_slice_lower",
]

import torch
from torch import Tensor

from ...utils import gen_slice_list


def batched_linear_backsub_slice(
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

    n_t, n_b = lower_symbnd_coeffs.shape[:2]
    shape = (n_t, n_b, *ori_shape)

    new_lower_symbnd_coeffs_tensor = torch.zeros(shape, **kwargs)
    new_upper_symbnd_coeffs_tensor = new_lower_symbnd_coeffs_tensor.clone()

    slicing = gen_slice_list(len(ori_shape), starts, ends, axes, steps)

    new_lower_symbnd_coeffs_tensor[:, :, *slicing] = lower_symbnd_coeffs
    new_upper_symbnd_coeffs_tensor[:, :, *slicing] = upper_symbnd_coeffs

    new_lower_symbnd_const = lower_symbnd_const
    new_upper_symbnd_const = upper_symbnd_const

    return (
        new_lower_symbnd_coeffs_tensor,
        new_lower_symbnd_const,
        new_upper_symbnd_coeffs_tensor,
        new_upper_symbnd_const,
    )


def batched_linear_backsub_slice_lower(
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

    n_t, n_b = lower_symbnd_coeffs.shape[0]
    shape = (n_t, n_b, *ori_shape)

    new_lower_symbnd_coeffs_tensor = torch.zeros(shape, **kwargs)

    slicing = gen_slice_list(len(ori_shape), starts, ends, axes, steps)

    new_lower_symbnd_coeffs_tensor[:, :, *slicing] = lower_symbnd_coeffs

    new_lower_symbnd_const = lower_symbnd_const

    return new_lower_symbnd_coeffs_tensor, new_lower_symbnd_const
