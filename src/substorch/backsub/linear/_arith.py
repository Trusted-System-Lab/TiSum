__all__ = [
    "linear_backsub_addcst",
    "linear_backsub_addcst_lower",
    "linear_backsub_subcst",
    "linear_backsub_subcst_lower",
    "linear_backsub_mulcst",
    "linear_backsub_mulcst_lower",
    "linear_backsub_divcst",
    "linear_backsub_divcst_lower",
    "linear_backsub_neg",
    "linear_backsub_neg_lower",
    "linear_backsub_add",
    "linear_backsub_add_lower",
    "linear_backsub_sub",
    "linear_backsub_sub_lower",
    "linear_backsub_mul",
    "linear_backsub_mul_lower",
    "linear_backsub_div",
    "linear_backsub_div_lower",
]

import torch
from torch import Tensor


def linear_backsub_addcst(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    constant: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    new_lower_symbnd_coeffs = lower_symbnd_coeffs
    new_upper_symbnd_coeffs = upper_symbnd_coeffs
    d = tuple(range(1, lower_symbnd_coeffs.dim()))
    new_lower_symbnd_const = lower_symbnd_const + (lower_symbnd_coeffs * constant).sum(
        dim=d
    )
    new_upper_symbnd_const = upper_symbnd_const + (upper_symbnd_coeffs * constant).sum(
        dim=d
    )

    return (
        new_lower_symbnd_coeffs,
        new_lower_symbnd_const,
        new_upper_symbnd_coeffs,
        new_upper_symbnd_const,
    )


def linear_backsub_addcst_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    constant: Tensor,
) -> tuple[Tensor, Tensor]:
    new_lower_symbnd_coeffs = lower_symbnd_coeffs
    d = tuple(range(1, lower_symbnd_coeffs.dim()))
    new_lower_symbnd_const = lower_symbnd_const + (lower_symbnd_coeffs * constant).sum(
        dim=d
    )

    return new_lower_symbnd_coeffs, new_lower_symbnd_const


def linear_backsub_subcst(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    constant: Tensor,
    is_left_op: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    d = tuple(range(1, lower_symbnd_coeffs.dim()))

    if is_left_op:
        new_lower_symbnd_coeffs = lower_symbnd_coeffs
        new_upper_symbnd_coeffs = upper_symbnd_coeffs
        new_lower_symbnd_const = lower_symbnd_const - (
            lower_symbnd_coeffs * constant
        ).sum(dim=d)
        new_upper_symbnd_const = upper_symbnd_const - (
            upper_symbnd_coeffs * constant
        ).sum(dim=d)
    else:
        new_lower_symbnd_coeffs = -lower_symbnd_coeffs
        new_upper_symbnd_coeffs = -upper_symbnd_coeffs
        new_lower_symbnd_const = lower_symbnd_const + (
            lower_symbnd_coeffs * constant
        ).sum(dim=d)
        new_upper_symbnd_const = upper_symbnd_const + (
            upper_symbnd_coeffs * constant
        ).sum(dim=d)

    return (
        new_lower_symbnd_coeffs,
        new_lower_symbnd_const,
        new_upper_symbnd_coeffs,
        new_upper_symbnd_const,
    )


def linear_backsub_subcst_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    constant: Tensor,
    is_left_op: bool = True,
) -> tuple[Tensor, Tensor]:
    d = tuple(range(1, lower_symbnd_coeffs.dim()))
    if is_left_op:
        new_lower_symbnd_coeffs = lower_symbnd_coeffs
        new_lower_symbnd_const = lower_symbnd_const - (
            lower_symbnd_coeffs * constant
        ).sum(dim=d)
    else:
        new_lower_symbnd_coeffs = -lower_symbnd_coeffs
        new_lower_symbnd_const = lower_symbnd_const + (
            lower_symbnd_coeffs * constant
        ).sum(dim=d)

    return new_lower_symbnd_coeffs, new_lower_symbnd_const


def linear_backsub_mulcst(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    constant: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    new_lower_symbnd_coeffs = lower_symbnd_coeffs * constant
    new_upper_symbnd_coeffs = upper_symbnd_coeffs * constant
    new_lower_symbnd_const = lower_symbnd_const
    new_upper_symbnd_const = upper_symbnd_const

    return (
        new_lower_symbnd_coeffs,
        new_lower_symbnd_const,
        new_upper_symbnd_coeffs,
        new_upper_symbnd_const,
    )


def linear_backsub_mulcst_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    constant: Tensor,
) -> tuple[Tensor, Tensor]:
    new_lower_symbnd_coeffs = lower_symbnd_coeffs * constant
    new_lower_symbnd_const = lower_symbnd_const

    return new_lower_symbnd_coeffs, new_lower_symbnd_const


def linear_backsub_divcst(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    constant: Tensor,
    is_left_op: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not is_left_op:
        raise NotImplementedError("Right division is not implemented yet. ")

    new_lower_symbnd_coeffs = lower_symbnd_coeffs / constant
    new_upper_symbnd_coeffs = upper_symbnd_coeffs / constant
    new_lower_symbnd_const = lower_symbnd_const
    new_upper_symbnd_const = upper_symbnd_const

    return (
        new_lower_symbnd_coeffs,
        new_lower_symbnd_const,
        new_upper_symbnd_coeffs,
        new_upper_symbnd_const,
    )


def linear_backsub_divcst_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    constant: Tensor,
    is_left_op: bool = True,
) -> tuple[Tensor, Tensor]:
    if not is_left_op:
        raise NotImplementedError("Right division is not implemented yet. ")

    new_lower_symbnd_coeffs = lower_symbnd_coeffs / constant
    new_lower_symbnd_const = lower_symbnd_const

    return new_lower_symbnd_coeffs, new_lower_symbnd_const


def linear_backsub_neg(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    new_lower_symbnd_coeffs = -lower_symbnd_coeffs
    new_upper_symbnd_coeffs = -upper_symbnd_coeffs
    new_lower_symbnd_const = lower_symbnd_const
    new_upper_symbnd_const = upper_symbnd_const

    return (
        new_lower_symbnd_coeffs,
        new_lower_symbnd_const,
        new_upper_symbnd_coeffs,
        new_upper_symbnd_const,
    )


def linear_backsub_neg_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
) -> tuple[Tensor, Tensor]:
    new_lower_symbnd_coeffs = -lower_symbnd_coeffs
    new_lower_symbnd_const = lower_symbnd_const

    return new_lower_symbnd_coeffs, new_lower_symbnd_const


def linear_backsub_add(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    lower_symbnd_coeffs1 = lower_symbnd_coeffs
    upper_symbnd_coeffs1 = upper_symbnd_coeffs
    lower_symbnd_const1 = lower_symbnd_const
    upper_symbnd_const1 = upper_symbnd_const
    lower_symbnd_coeffs2 = lower_symbnd_coeffs
    upper_symbnd_coeffs2 = upper_symbnd_coeffs
    lower_symbnd_const2 = torch.zeros_like(lower_symbnd_const)
    upper_symbnd_const2 = torch.zeros_like(upper_symbnd_const)

    return (
        lower_symbnd_coeffs1,
        lower_symbnd_const1,
        lower_symbnd_coeffs2,
        lower_symbnd_const2,
        upper_symbnd_coeffs1,
        upper_symbnd_const1,
        upper_symbnd_coeffs2,
        upper_symbnd_const2,
    )


def linear_backsub_add_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    lower_symbnd_coeffs1 = lower_symbnd_coeffs
    lower_symbnd_const1 = lower_symbnd_const
    lower_symbnd_coeffs2 = lower_symbnd_coeffs
    lower_symbnd_const2 = torch.zeros_like(lower_symbnd_const)

    return (
        lower_symbnd_coeffs1,
        lower_symbnd_const1,
        lower_symbnd_coeffs2,
        lower_symbnd_const2,
    )


def linear_backsub_sub(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    lower_symbnd_coeffs1 = lower_symbnd_coeffs
    upper_symbnd_coeffs1 = upper_symbnd_coeffs
    lower_symbnd_const1 = lower_symbnd_const
    upper_symbnd_const1 = upper_symbnd_const
    lower_symbnd_coeffs2 = -lower_symbnd_coeffs
    upper_symbnd_coeffs2 = -upper_symbnd_coeffs
    lower_symbnd_const2 = torch.zeros_like(lower_symbnd_const)
    upper_symbnd_const2 = torch.zeros_like(upper_symbnd_const)

    return (
        lower_symbnd_coeffs1,
        lower_symbnd_const1,
        lower_symbnd_coeffs2,
        lower_symbnd_const2,
        upper_symbnd_coeffs1,
        upper_symbnd_const1,
        upper_symbnd_coeffs2,
        upper_symbnd_const2,
    )


def linear_backsub_sub_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    lower_symbnd_coeffs1 = lower_symbnd_coeffs
    lower_symbnd_const1 = -lower_symbnd_const
    lower_symbnd_coeffs2 = -lower_symbnd_coeffs
    lower_symbnd_const2 = torch.zeros_like(lower_symbnd_const)

    return (
        lower_symbnd_coeffs1,
        lower_symbnd_const1,
        lower_symbnd_coeffs2,
        lower_symbnd_const2,
    )


def _linear_backsub_binary(
    symbnd_coeffs: Tensor,
    symbnd_const: Tensor,
    rlx_coeffs_a1: Tensor,
    rlx_coeffs_b1: Tensor,
    rlx_const1: Tensor,
    rlx_coeffs_a2: Tensor,
    rlx_coeffs_b2: Tensor,
    rlx_const2: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    pos_symbnd_coeffs = symbnd_coeffs.clamp_min(0)
    neg_symbnd_coeffs = symbnd_coeffs.clamp_max(0)

    d = tuple(range(1, symbnd_coeffs.dim()))
    new_symbnd_cst_a = (
        symbnd_const
        + (pos_symbnd_coeffs * rlx_const1).sum(dim=d)
        + (neg_symbnd_coeffs * rlx_const2).sum(dim=d)
    )
    new_symbnd_cst_b = torch.zeros_like(new_symbnd_cst_a)
    new_symbnd_coeffs_a = (
        pos_symbnd_coeffs * rlx_coeffs_a1 + neg_symbnd_coeffs * rlx_coeffs_a2
    )
    new_symbnd_coeffs_b = (
        pos_symbnd_coeffs * rlx_coeffs_b1 + neg_symbnd_coeffs * rlx_coeffs_b2
    )

    return (
        new_symbnd_coeffs_a,
        new_symbnd_cst_a,
        new_symbnd_coeffs_b,
        new_symbnd_cst_b,
    )


def linear_backsub_binary(
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
    (
        new_lower_symbnd_coeffs_a,
        new_lower_symbnd_const_a,
        new_lower_symbnd_coeffs_b,
        new_lower_symbnd_const_b,
    ) = _linear_backsub_binary(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs_a,
        lower_rlx_coeffs_b,
        lower_rlx_const,
        upper_rlx_coeffs_a,
        upper_rlx_coeffs_b,
        upper_rlx_const,
    )

    (
        new_upper_symbnd_coeffs_a,
        new_upper_symbnd_const_a,
        new_upper_symbnd_coeffs_b,
        new_upper_symbnd_const_b,
    ) = _linear_backsub_binary(
        upper_symbnd_coeffs,
        upper_symbnd_const,
        upper_rlx_coeffs_a,
        upper_rlx_coeffs_b,
        upper_rlx_const,
        lower_rlx_coeffs_a,
        lower_rlx_coeffs_b,
        lower_rlx_const,
    )

    return (
        new_lower_symbnd_coeffs_a,
        new_lower_symbnd_const_a,
        new_lower_symbnd_coeffs_b,
        new_lower_symbnd_const_b,
        new_upper_symbnd_coeffs_a,
        new_upper_symbnd_const_a,
        new_upper_symbnd_coeffs_b,
        new_upper_symbnd_const_b,
    )


def linear_backsub_binary_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs_a: Tensor,
    lower_rlx_coeffs_b: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs_a: Tensor,
    upper_rlx_coeffs_b: Tensor,
    upper_rlx_const: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _linear_backsub_binary(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs_a,
        lower_rlx_coeffs_b,
        lower_rlx_const,
        upper_rlx_coeffs_a,
        upper_rlx_coeffs_b,
        upper_rlx_const,
    )


def linear_backsub_mul(
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
    return linear_backsub_binary(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        lower_rlx_coeffs_a,
        lower_rlx_coeffs_b,
        lower_rlx_const,
        upper_rlx_coeffs_a,
        upper_rlx_coeffs_b,
        upper_rlx_const,
    )


def linear_backsub_mul_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs_a: Tensor,
    lower_rlx_coeffs_b: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs_a: Tensor,
    upper_rlx_coeffs_b: Tensor,
    upper_rlx_const: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_binary_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs_a,
        lower_rlx_coeffs_b,
        lower_rlx_const,
        upper_rlx_coeffs_a,
        upper_rlx_coeffs_b,
        upper_rlx_const,
    )


def linear_backsub_div(
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
    return linear_backsub_binary(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        lower_rlx_coeffs_a,
        lower_rlx_coeffs_b,
        lower_rlx_const,
        upper_rlx_coeffs_a,
        upper_rlx_coeffs_b,
        upper_rlx_const,
    )


def linear_backsub_div_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs_a: Tensor,
    lower_rlx_coeffs_b: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs_a: Tensor,
    upper_rlx_coeffs_b: Tensor,
    upper_rlx_const: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_binary_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs_a,
        lower_rlx_coeffs_b,
        lower_rlx_const,
        upper_rlx_coeffs_a,
        upper_rlx_coeffs_b,
        upper_rlx_const,
    )
