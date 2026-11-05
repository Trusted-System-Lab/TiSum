__all__ = [
    "linear_backsub_concat",
    "linear_backsub_concat_lower",
    "linear_backsub_split",
    "linear_backsub_split_lower",
]

import torch
from torch import Tensor


def linear_backsub_concat(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    split_size_or_sections: int | list[int],
    dim: int = 0,
) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
    dim = dim + 1
    new_lower_symbnd_coeffs_list = list(
        torch.split(lower_symbnd_coeffs, split_size_or_sections, dim=dim)
    )
    new_upper_symbnd_coeffs_list = list(
        torch.split(upper_symbnd_coeffs, split_size_or_sections, dim=dim)
    )

    new_lower_symbnd_const_list = [lower_symbnd_const]
    new_upper_symbnd_const_list = [upper_symbnd_const]
    for i in range(1, len(new_lower_symbnd_coeffs_list)):
        zeros = torch.zeros_like(lower_symbnd_const)
        new_lower_symbnd_const_list.append(zeros)
        new_upper_symbnd_const_list.append(zeros)

    return (
        new_lower_symbnd_coeffs_list,
        new_lower_symbnd_const_list,
        new_upper_symbnd_coeffs_list,
        new_upper_symbnd_const_list,
    )


def linear_backsub_concat_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    split_size_or_sections: list[int],
    dim: int = 0,
) -> tuple[list[Tensor], list[Tensor]]:
    dim = dim + 1
    new_lower_symbnd_coeffs_list = list(
        torch.split(lower_symbnd_coeffs, split_size_or_sections, dim=dim)
    )

    new_lower_symbnd_const_list = [lower_symbnd_const]
    for i in range(1, len(new_lower_symbnd_coeffs_list)):
        zeros = torch.zeros_like(lower_symbnd_const)
        new_lower_symbnd_const_list.append(zeros)

    return new_lower_symbnd_coeffs_list, new_lower_symbnd_const_list


def linear_backsub_split(
    lower_symbnd_coeffs_list: list[Tensor],
    lower_symbnd_const_list: list[Tensor],
    upper_symbnd_coeffs_list: list[Tensor],
    upper_symbnd_const_list: list[Tensor],
    *,
    dim: int = 0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    dim = dim + 1
    new_lower_symbnd_coeffs = torch.cat(lower_symbnd_coeffs_list, dim=dim)
    new_upper_symbnd_coeffs = torch.cat(upper_symbnd_coeffs_list, dim=dim)
    new_lower_symbnd_const = torch.stack(lower_symbnd_const_list).sum(dim=0)
    new_upper_symbnd_const = torch.stack(upper_symbnd_const_list).sum(dim=0)

    return (
        new_lower_symbnd_coeffs,
        new_lower_symbnd_const,
        new_upper_symbnd_coeffs,
        new_upper_symbnd_const,
    )


def linear_backsub_split_lower(
    lower_symbnd_coeffs_list: list[Tensor],
    lower_symbnd_const_list: list[Tensor],
    *,
    dim: int = 0,
) -> tuple[Tensor, Tensor]:
    dim = dim + 1
    new_lower_symbnd_coeffs = torch.cat(lower_symbnd_coeffs_list, dim=dim)
    new_lower_symbnd_const = torch.stack(lower_symbnd_const_list).sum(dim=0)

    return new_lower_symbnd_coeffs, new_lower_symbnd_const
