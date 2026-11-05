__all__ = ["ada_backsub_relu", "ada_backsub_relu_lower"]

from torch import Tensor

from ..linear import linear_backsub_unary, linear_backsub_unary_lower


def ada_backsub_relu(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    lower_ada_idx: Tensor,
    lower_ada_param: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if lower_ada_param is not None:
        lower_rlx_coeffs[lower_ada_idx] = lower_ada_param

    return linear_backsub_unary(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
    )


def ada_backsub_relu_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    lower_ada_idx: Tensor,
    lower_ada_param: Tensor | None,
) -> tuple[Tensor, Tensor]:
    if lower_ada_param is not None:
        lower_rlx_coeffs[lower_ada_idx] = lower_ada_param

    return linear_backsub_unary_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
    )
