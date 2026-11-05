__all__ = ["batched_ada_backsub_relu", "batched_ada_backsub_relu_lower"]

from torch import Tensor

from ..batched_linear import (
    batched_linear_backsub_unary,
    batched_linear_backsub_unary_lower,
)


def batched_ada_backsub_relu(
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
        assert lower_ada_idx.dim() == 2, f"{lower_ada_idx.shape}\n{lower_ada_idx}"
        lower_rlx_coeffs[tuple(lower_ada_idx.T)] = lower_ada_param

    return batched_linear_backsub_unary(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
    )


def batched_ada_backsub_relu_lower(
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
        lower_rlx_coeffs[tuple(lower_ada_idx.T)] = lower_ada_param

    return batched_linear_backsub_unary_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
    )
