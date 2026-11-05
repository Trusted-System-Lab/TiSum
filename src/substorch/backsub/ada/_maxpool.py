__all__ = ["ada_backsub_maxpool2d", "ada_backsub_maxpool2d_lower"]

from torch import Tensor

from ..linear import linear_backsub_maxpool2d, linear_backsub_maxpool2d_lower


def ada_backsub_maxpool2d(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    input_size: tuple[int, int, int],
    output_size: tuple[int, int, int],
    kernel_size: tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    lower_ada_idx: Tensor,
    lower_ada_param: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if lower_ada_param is not None:
        lower_rlx_coeffs[lower_ada_idx] = lower_ada_param

    return linear_backsub_maxpool2d(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        input_size=input_size,
        output_size=output_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )


def ada_backsub_maxpool2d_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    input_size: tuple[int, int, int],
    output_size: tuple[int, int, int],
    kernel_size: tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    lower_ada_idx: Tensor,
    lower_ada_param: Tensor,
) -> tuple[Tensor, Tensor]:
    if lower_ada_param is not None:
        lower_rlx_coeffs[lower_ada_idx] = lower_ada_param

    return linear_backsub_maxpool2d_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        input_size=input_size,
        output_size=output_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
