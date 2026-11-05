__all__ = [
    "linear_backsub_expand",
    "linear_backsub_expand_lower",
    "linear_backsub_flatten",
    "linear_backsub_flatten_lower",
    "linear_backsub_permute",
    "linear_backsub_permute_lower",
    "linear_backsub_reshape",
    "linear_backsub_reshape_lower",
    "linear_backsub_squeeze",
    "linear_backsub_squeeze_lower",
    "linear_backsub_unsqueeze",
    "linear_backsub_unsqueeze_lower",
]

import torch
from torch import Tensor


def linear_backsub_expand(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    ori_shape: tuple[int, ...],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    one_dims = [i for i, dim in enumerate(ori_shape) if dim == 1]
    lower_symbnd_coeffs = torch.sum(lower_symbnd_coeffs, dim=one_dims)
    upper_symbnd_coeffs = torch.sum(upper_symbnd_coeffs, dim=one_dims)

    lower_symbnd_const = lower_symbnd_const
    upper_symbnd_const = upper_symbnd_const

    return (
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
    )


def linear_backsub_expand_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    ori_shape: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    one_dims = [i for i, dim in enumerate(ori_shape) if dim == 1]
    lower_symbnd_coeffs = torch.sum(lower_symbnd_coeffs, dim=one_dims)

    lower_symbnd_const = lower_symbnd_const

    return lower_symbnd_coeffs, lower_symbnd_const


def linear_backsub_flatten(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    ori_shape: tuple[int, ...],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    lower_symbnd_coeffs = lower_symbnd_coeffs.reshape(-1, *ori_shape)
    upper_symbnd_coeffs = upper_symbnd_coeffs.reshape(-1, *ori_shape)

    return (
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
    )


def linear_backsub_flatten_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    ori_shape: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    lower_symbnd_coeffs = lower_symbnd_coeffs.reshape(-1, *ori_shape)

    return lower_symbnd_coeffs, lower_symbnd_const


def linear_backsub_permute(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    dims: tuple[int, ...],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    dims = (0,) + tuple(d + 1 for d in dims)
    lower_symbnd_coeffs = lower_symbnd_coeffs.permute(dims)
    upper_symbnd_coeffs = upper_symbnd_coeffs.permute(dims)
    lower_symbnd_const = lower_symbnd_const
    upper_symbnd_const = upper_symbnd_const

    return (
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
    )


def linear_backsub_permute_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    dims: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    lower_symbnd_coeffs = lower_symbnd_coeffs.permute(dims)
    lower_symbnd_const = lower_symbnd_const

    return lower_symbnd_coeffs, lower_symbnd_const


def linear_backsub_reshape(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    ori_shape: tuple[int, ...],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_flatten(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        ori_shape=ori_shape,
    )


def linear_backsub_reshape_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    ori_shape: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    return linear_backsub_flatten_lower(
        lower_symbnd_coeffs, lower_symbnd_const, ori_shape=ori_shape
    )


def linear_backsub_squeeze(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    dim: int | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    lower_symbnd_coeffs = lower_symbnd_coeffs.unsqueeze(dim)
    upper_symbnd_coeffs = upper_symbnd_coeffs.unsqueeze(dim)
    lower_symbnd_const = lower_symbnd_const
    upper_symbnd_const = upper_symbnd_const

    return (
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
    )


def linear_backsub_squeeze_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    dim: int | None = None,
) -> tuple[Tensor, Tensor]:
    lower_symbnd_coeffs = lower_symbnd_coeffs.unsqueeze(dim)
    lower_symbnd_const = lower_symbnd_const

    return lower_symbnd_coeffs, lower_symbnd_const


def linear_backsub_unsqueeze(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    dim: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    lower_symbnd_coeffs = lower_symbnd_coeffs.squeeze(dim)
    upper_symbnd_coeffs = upper_symbnd_coeffs.squeeze(dim)
    lower_symbnd_const = lower_symbnd_const
    upper_symbnd_const = upper_symbnd_const

    return (
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
    )


def linear_backsub_unsqueeze_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    dim: int,
) -> tuple[Tensor, Tensor]:
    lower_symbnd_coeffs = lower_symbnd_coeffs.squeeze(dim)
    lower_symbnd_const = lower_symbnd_const

    return lower_symbnd_coeffs, lower_symbnd_const
