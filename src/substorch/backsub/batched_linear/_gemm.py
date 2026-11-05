__all__ = ["batched_linear_backsub_gemm", "batched_linear_backsub_gemm_lower"]

from torch import Tensor, einsum


def _batched_linear_backsub_gemm(
    symbnd_coeffs: Tensor,
    symbnd_const: Tensor,
    *,
    weight: Tensor,
    bias: Tensor,
) -> tuple[Tensor, Tensor]:
    new_symbnd_const = symbnd_const + einsum("abc, c -> ab", symbnd_coeffs, bias)
    new_symbnd_coeffs = einsum("abc, cd -> abd", symbnd_coeffs, weight)

    return new_symbnd_coeffs, new_symbnd_const


def batched_linear_backsub_gemm(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    weight: Tensor,
    bias: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    weight = weight.t()

    new_lower_symbnd_coeffs, new_lower_symbnd_const = _batched_linear_backsub_gemm(
        lower_symbnd_coeffs, lower_symbnd_const, weight=weight, bias=bias
    )
    new_upper_symbnd_coeffs, new_upper_symbnd_const = _batched_linear_backsub_gemm(
        upper_symbnd_coeffs, upper_symbnd_const, weight=weight, bias=bias
    )

    return (
        new_lower_symbnd_coeffs,
        new_lower_symbnd_const,
        new_upper_symbnd_coeffs,
        new_upper_symbnd_const,
    )


def batched_linear_backsub_gemm_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    weight: Tensor,
    bias: Tensor,
) -> tuple[Tensor, Tensor]:
    weight = weight.t()

    return _batched_linear_backsub_gemm(
        lower_symbnd_coeffs, lower_symbnd_const, weight=weight, bias=bias
    )
