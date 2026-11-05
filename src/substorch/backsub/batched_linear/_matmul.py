__all__ = [
    "batched_linear_backsub_matmulcst",
    "batched_linear_backsub_matmulcst_lower",
]

from torch import Tensor


def _batched_linear_backsub_matmulcst(
    symbnd_coeffs: Tensor,
    symbnd_const: Tensor,
    *,
    weight: Tensor,
) -> tuple[Tensor, Tensor]:
    new_symbnd_const = symbnd_const
    new_symbnd_coeffs = symbnd_coeffs @ weight

    return new_symbnd_coeffs, new_symbnd_const


def batched_linear_backsub_matmulcst(
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

    new_lower_symbnd_coeffs, new_lower_symbnd_const = _batched_linear_backsub_matmulcst(
        lower_symbnd_coeffs, lower_symbnd_const, weight=weight
    )
    new_upper_symbnd_coeffs, new_upper_symbnd_const = _batched_linear_backsub_matmulcst(
        upper_symbnd_coeffs, upper_symbnd_const, weight=weight
    )

    return (
        new_lower_symbnd_coeffs,
        new_lower_symbnd_const,
        new_upper_symbnd_coeffs,
        new_upper_symbnd_const,
    )


def batched_linear_backsub_matmulcst_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    weight: Tensor,
    is_left_op: bool = True,
) -> tuple[Tensor, Tensor]:
    if not is_left_op:
        weight = weight.t()

    return _batched_linear_backsub_matmulcst(
        lower_symbnd_coeffs, lower_symbnd_const, weight=weight
    )
