__all__ = ["batched_linear_backsub_unary", "batched_linear_backsub_unary_lower"]

from torch import Tensor


def _batched_linear_backsub_unary(
    symbnd_coeffs: Tensor,
    symbnd_const: Tensor,
    rlx_coeffs1: Tensor,
    rlx_const1: Tensor,
    rlx_coeffs2: Tensor,
    rlx_const2: Tensor,
) -> tuple[Tensor, Tensor]:
    pos_symbnd_coeffs = symbnd_coeffs.clamp_min(0)
    neg_symbnd_coeffs = symbnd_coeffs.clamp_max(0)

    d = list(range(2, symbnd_coeffs.dim()))
    new_symbnd_const = (
        symbnd_const
        + (pos_symbnd_coeffs * rlx_const1).sum(dim=d)
        + (neg_symbnd_coeffs * rlx_const2).sum(dim=d)
    )
    new_symbnd_coeffs = (
        pos_symbnd_coeffs * rlx_coeffs1 + neg_symbnd_coeffs * rlx_coeffs2
    )

    return new_symbnd_coeffs, new_symbnd_const


def batched_linear_backsub_unary(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    lower_rlx_coeffs = lower_rlx_coeffs.unsqueeze(1)
    lower_rlx_const = lower_rlx_const.unsqueeze(1)
    upper_rlx_coeffs = upper_rlx_coeffs.unsqueeze(1)
    upper_rlx_const = upper_rlx_const.unsqueeze(1)

    new_lower_symbnd_coeffs, new_lower_symbnd_const = _batched_linear_backsub_unary(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
    )
    new_upper_symbnd_coeffs, new_upper_symbnd_const = _batched_linear_backsub_unary(
        upper_symbnd_coeffs,
        upper_symbnd_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        lower_rlx_coeffs,
        lower_rlx_const,
    )

    return (
        new_lower_symbnd_coeffs,
        new_lower_symbnd_const,
        new_upper_symbnd_coeffs,
        new_upper_symbnd_const,
    )


def batched_linear_backsub_unary_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
) -> tuple[Tensor, Tensor]:
    lower_rlx_coeffs = lower_rlx_coeffs.unsqueeze(1)
    lower_rlx_const = lower_rlx_const.unsqueeze(1)
    upper_rlx_coeffs = upper_rlx_coeffs.unsqueeze(1)
    upper_rlx_const = upper_rlx_const.unsqueeze(1)

    new_lower_symbnd_coeffs, new_lower_symbnd_const = _batched_linear_backsub_unary(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
    )

    return new_lower_symbnd_coeffs, new_lower_symbnd_const
