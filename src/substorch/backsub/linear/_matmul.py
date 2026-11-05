__all__ = [
    "linear_backsub_matmul",
    "linear_backsub_matmul_lower",
    "linear_backsub_matmulcst",
    "linear_backsub_matmulcst_lower",
]

import torch
from torch import Tensor


def linear_backsub_matmul(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs_a: Tensor,
    lower_rlx_coeffs_b: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs_a: Tensor,
    upper_rlx_coeffs_b: Tensor,
    upper_rlx_const: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    lower_symbnd_coeffs = lower_symbnd_coeffs.unsqueeze(2)
    upper_symbnd_coeffs = upper_symbnd_coeffs.unsqueeze(2)
    lower_rlx_coeffs_a = lower_rlx_coeffs_a.unsqueeze(0)
    lower_rlx_coeffs_b = lower_rlx_coeffs_b.unsqueeze(0)
    lower_rlx_const = lower_rlx_const.unsqueeze(0).unsqueeze(-2)
    upper_rlx_coeffs_a = upper_rlx_coeffs_a.unsqueeze(0)
    upper_rlx_coeffs_b = upper_rlx_coeffs_b.unsqueeze(0)
    upper_rlx_const = upper_rlx_const.unsqueeze(0).unsqueeze(-2)

    pos_lower_symbnd_coeffs = lower_symbnd_coeffs.clamp_min(0)
    neg_lower_symbnd_coeffs = lower_symbnd_coeffs.clamp_max(0)

    new_lower_symbnd_coeffs_a = (
        pos_lower_symbnd_coeffs * lower_rlx_coeffs_a
        + neg_lower_symbnd_coeffs * upper_rlx_coeffs_a
    ).sum(dim=-1)
    new_lower_symbnd_coeffs_b = (
        pos_lower_symbnd_coeffs * lower_rlx_coeffs_b
        + neg_lower_symbnd_coeffs * upper_rlx_coeffs_b
    ).sum(dim=-1)

    new_lower_symbnd_const_a = (
        lower_symbnd_const
        + (pos_lower_symbnd_coeffs * lower_rlx_const).sum(dim=(1, 2, 3))
        + (neg_lower_symbnd_coeffs * upper_rlx_const).sum(dim=(1, 2, 3))
    )
    new_upper_symbnd_const_a = (
        upper_symbnd_const
        + (pos_lower_symbnd_coeffs * lower_rlx_const).sum(dim=(1, 2, 3))
        + (neg_lower_symbnd_coeffs * upper_rlx_const).sum(dim=(1, 2, 3))
    )

    pos_upper_symbnd_coeffs = upper_symbnd_coeffs.clamp_min(0)
    neg_upper_symbnd_coeffs = upper_symbnd_coeffs.clamp_max(0)

    new_upper_symbnd_coeffs_a = (
        pos_upper_symbnd_coeffs * lower_rlx_coeffs_a
        + neg_upper_symbnd_coeffs * upper_rlx_coeffs_a
    ).sum(dim=1)
    new_upper_symbnd_coeffs_b = (
        pos_upper_symbnd_coeffs * lower_rlx_coeffs_b
        + neg_upper_symbnd_coeffs * upper_rlx_coeffs_b
    ).sum(dim=1)

    new_lower_symbnd_const_b = torch.zeros_like(new_lower_symbnd_const_a)
    new_upper_symbnd_const_b = torch.zeros_like(new_upper_symbnd_const_a)

    return (
        new_lower_symbnd_coeffs_a,
        new_lower_symbnd_coeffs_b,
        new_lower_symbnd_const_a,
        new_lower_symbnd_const_b,
        new_upper_symbnd_coeffs_a,
        new_upper_symbnd_coeffs_b,
        new_upper_symbnd_const_a,
        new_upper_symbnd_const_b,
    )


def linear_backsub_matmul_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs_a: Tensor,
    lower_rlx_coeffs_b: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs_a: Tensor,
    upper_rlx_coeffs_b: Tensor,
    upper_rlx_const: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    lower_symbnd_coeffs = lower_symbnd_coeffs.unsqueeze(2)
    lower_rlx_coeffs_a = lower_rlx_coeffs_a.unsqueeze(0)
    lower_rlx_coeffs_b = lower_rlx_coeffs_b.unsqueeze(0)
    lower_rlx_const = lower_rlx_const.unsqueeze(0).unsqueeze(-2)
    upper_rlx_coeffs_a = upper_rlx_coeffs_a.unsqueeze(0)
    upper_rlx_coeffs_b = upper_rlx_coeffs_b.unsqueeze(0)
    upper_rlx_const = upper_rlx_const.unsqueeze(0).unsqueeze(-2)

    pos_lower_symbnd_coeffs = lower_symbnd_coeffs.clamp_min(0)
    neg_lower_symbnd_coeffs = lower_symbnd_coeffs.clamp_max(0)

    new_lower_symbnd_coeffs_a = (
        pos_lower_symbnd_coeffs * lower_rlx_coeffs_a
        + neg_lower_symbnd_coeffs * upper_rlx_coeffs_a
    ).sum(dim=-1)
    new_lower_symbnd_coeffs_b = (
        pos_lower_symbnd_coeffs * lower_rlx_coeffs_b
        + neg_lower_symbnd_coeffs * upper_rlx_coeffs_b
    ).sum(dim=-1)

    new_lower_symbnd_const_a = (
        lower_symbnd_const
        + (pos_lower_symbnd_coeffs * lower_rlx_const).sum(dim=(1, 2, 3))
        + (neg_lower_symbnd_coeffs * upper_rlx_const).sum(dim=(1, 2, 3))
    )

    new_lower_symbnd_const_b = torch.zeros_like(new_lower_symbnd_const_a)

    return (
        new_lower_symbnd_coeffs_a,
        new_lower_symbnd_coeffs_b,
        new_lower_symbnd_const_a,
        new_lower_symbnd_const_b,
    )


def _linear_backsub_matmulcst(
    symbnd_coeffs: Tensor,
    symbnd_const: Tensor,
    *,
    weight: Tensor,
) -> tuple[Tensor, Tensor]:
    new_symbnd_const = symbnd_const
    new_symbnd_coeffs = symbnd_coeffs @ weight

    return new_symbnd_coeffs, new_symbnd_const


def linear_backsub_matmulcst(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    weight: Tensor,
    is_left_op: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not is_left_op:
        weight = weight.t()

    new_lower_symbnd_coeffs, new_lower_symbnd_const = _linear_backsub_matmulcst(
        lower_symbnd_coeffs, lower_symbnd_const, weight=weight
    )
    new_upper_symbnd_coeffs, new_upper_symbnd_const = _linear_backsub_matmulcst(
        upper_symbnd_coeffs, upper_symbnd_const, weight=weight
    )

    return (
        new_lower_symbnd_coeffs,
        new_lower_symbnd_const,
        new_upper_symbnd_coeffs,
        new_upper_symbnd_const,
    )


def linear_backsub_matmulcst_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    weight: Tensor,
    is_left_op: bool = True,
) -> tuple[Tensor, Tensor]:
    if not is_left_op:
        weight = weight.t()

    return _linear_backsub_matmulcst(
        lower_symbnd_coeffs, lower_symbnd_const, weight=weight
    )
