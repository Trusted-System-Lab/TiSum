__all__ = ["quad_backsub_unary", "quad_backsub_unary_lower"]

from torch import Tensor


def _quad_backsub_unary(
    symbnd_coeffs: Tensor,
    symbnd_const: Tensor,
    rlx_quad_coeffs1: Tensor,
    rlx_linear_coeffs1: Tensor,
    rlx_const1: Tensor,
    rlx_quad_coeffs2: Tensor,
    rlx_linear_coeffs2: Tensor,
    rlx_const2: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:

    pos_symbnd_coeffs = symbnd_coeffs.clamp_min(0)
    neg_symbnd_coeffs = symbnd_coeffs.clamp_max(0)

    d = list(range(1, symbnd_coeffs.dim()))
    new_symbnd_const = (
        symbnd_const
        + (pos_symbnd_coeffs * rlx_const1).sum(d)
        + (neg_symbnd_coeffs * rlx_const2).sum(d)
    )
    new_symbnd_linear_coeffs = (
        pos_symbnd_coeffs * rlx_linear_coeffs1 + neg_symbnd_coeffs * rlx_linear_coeffs2
    )
    new_symbnd_quad_coeffs = (
        pos_symbnd_coeffs * rlx_quad_coeffs1 + neg_symbnd_coeffs * rlx_quad_coeffs2
    )

    return new_symbnd_quad_coeffs, new_symbnd_linear_coeffs, new_symbnd_const


def quad_backsub_unary(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_quad_coeffs: Tensor,
    lower_rlx_linear_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_quad_coeffs: Tensor,
    upper_rlx_linear_coeffs: Tensor,
    upper_rlx_const: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    lower_rlx_quad_coeffs = lower_rlx_quad_coeffs.unsqueeze(0)
    lower_rlx_linear_coeffs = lower_rlx_linear_coeffs.unsqueeze(0)
    lower_rlx_const = lower_rlx_const.unsqueeze(0)
    upper_rlx_quad_coeffs = upper_rlx_quad_coeffs.unsqueeze(0)
    upper_rlx_linear_coeffs = upper_rlx_linear_coeffs.unsqueeze(0)
    upper_rlx_const = upper_rlx_const.unsqueeze(0)

    (
        new_lower_symbnd_quad_coeffs,
        new_lower_symbnd_linear_coeffs,
        new_lower_symbnd_const,
    ) = _quad_backsub_unary(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_quad_coeffs,
        lower_rlx_linear_coeffs,
        lower_rlx_const,
        upper_rlx_quad_coeffs,
        upper_rlx_linear_coeffs,
        upper_rlx_const,
    )
    (
        new_upper_symbnd_quad_coeffs,
        new_upper_symbnd_linear_coeffs,
        new_upper_symbnd_const,
    ) = _quad_backsub_unary(
        upper_symbnd_coeffs,
        upper_symbnd_const,
        upper_rlx_quad_coeffs,
        upper_rlx_linear_coeffs,
        upper_rlx_const,
        lower_rlx_quad_coeffs,
        lower_rlx_linear_coeffs,
        lower_rlx_const,
    )
    return (
        new_lower_symbnd_quad_coeffs,
        new_lower_symbnd_linear_coeffs,
        new_lower_symbnd_const,
        new_upper_symbnd_quad_coeffs,
        new_upper_symbnd_linear_coeffs,
        new_upper_symbnd_const,
    )


def quad_backsub_unary_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_quad_coeffs: Tensor,
    lower_rlx_linear_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_quad_coeffs: Tensor,
    upper_rlx_linear_coeffs: Tensor,
    upper_rlx_const: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    lower_rlx_quad_coeffs = lower_rlx_quad_coeffs.unsqueeze(0)
    lower_rlx_linear_coeffs = lower_rlx_linear_coeffs.unsqueeze(0)
    lower_rlx_const = lower_rlx_const.unsqueeze(0)
    upper_rlx_quad_coeffs = upper_rlx_quad_coeffs.unsqueeze(0)
    upper_rlx_linear_coeffs = upper_rlx_linear_coeffs.unsqueeze(0)
    upper_rlx_const = upper_rlx_const.unsqueeze(0)

    return _quad_backsub_unary(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_quad_coeffs,
        lower_rlx_linear_coeffs,
        lower_rlx_const,
        upper_rlx_quad_coeffs,
        upper_rlx_linear_coeffs,
        upper_rlx_const,
    )
