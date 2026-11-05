__all__ = [
    "batched_linear_backsub_concat",
    "batched_linear_backsub_concat_lower",
]

import torch
from torch import Tensor


def batched_linear_backsub_concat(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    split_size_or_sections: int | list[int],
    dim: int = 0,
) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
    dim = dim + 2
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


def batched_linear_backsub_concat_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    split_size_or_sections: list[int],
    dim: int = 0,
) -> tuple[list[Tensor], list[Tensor]]:
    dim = dim + 2
    new_lower_symbnd_coeffs_list = list(
        torch.split(lower_symbnd_coeffs, split_size_or_sections, dim=dim)
    )

    new_lower_symbnd_const_list = [lower_symbnd_const]
    for i in range(1, len(new_lower_symbnd_coeffs_list)):
        zeros = torch.zeros_like(lower_symbnd_const)
        new_lower_symbnd_const_list.append(zeros)

    return new_lower_symbnd_coeffs_list, new_lower_symbnd_const_list
