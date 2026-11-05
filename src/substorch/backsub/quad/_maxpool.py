__all__ = [
    "quad_backsub_maxpool2d_after_relu",
    "quad_backsub_maxpool2d_after_relu_lower",
]

import torch.nn.functional as F
from torch import Tensor


def _quad_backsub_maxpool2d_after_relu(
    A: Tensor,
    b: Tensor,
    rlx_quad_coeffs1: Tensor,
    rlx_linear_coeffs1: Tensor,
    rlx_const1: Tensor,
    rlx_quad_coeffs2: Tensor,
    rlx_linear_coeffs2: Tensor,
    rlx_const2: Tensor,
    *,
    input_size: tuple[int, int, int],
    output_size: tuple[int, int, int],
    kernel_size: tuple[int, int],
    dilation: int | tuple[int, int],
    padding: int | tuple[int, int],
    stride: int | tuple[int, int],
) -> tuple[Tensor, Tensor, Tensor]:
    Ap, An = A.clamp_min(0), A.clamp_max(0)
    b_linear = Ap @ rlx_const1 + An @ rlx_const2
    b = b + b_linear

    Ap = Ap.unsqueeze(-1)
    An = An.unsqueeze(-1)
    rlx_quad_coeffs1 = rlx_quad_coeffs1.unsqueeze(0)
    rlx_quad_coeffs2 = rlx_quad_coeffs2.unsqueeze(0)
    rlx_linear_coeffs1 = rlx_linear_coeffs1.unsqueeze(0)
    rlx_linear_coeffs2 = rlx_linear_coeffs2.unsqueeze(0)

    Q = Ap * rlx_quad_coeffs1 + An * rlx_quad_coeffs2
    A = Ap * rlx_linear_coeffs1 + An * rlx_linear_coeffs2

    c, h1, w1 = input_size
    h2, w2 = output_size[1:]
    hw2 = h2 * w2
    kk = kernel_size[0] * kernel_size[1]
    n_b = b.shape[0]
    Q = Q.reshape(n_b, c, hw2, kk).permute(0, 1, 3, 2).reshape(n_b * c, kk, hw2)
    A = A.reshape(n_b, c, hw2, kk).permute(0, 1, 3, 2).reshape(n_b * c, kk, hw2)

    args = (input_size[1:], kernel_size, dilation, padding, stride)
    Q = F.fold(Q, *args)
    A = F.fold(A, *args)
    Q = Q.reshape(n_b, c * h1 * w1)
    A = A.reshape(n_b, c * h1 * w1)

    return (
        Q.reshape(-1, *input_size),
        A.reshape(-1, *input_size),
        b.reshape(-1),
    )


def quad_backsub_maxpool2d_after_relu(
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
    *,
    input_size: tuple[int, int, int],
    output_size: tuple[int, int, int],
    kernel_size: tuple[int, int],
    dilation: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    stride: int | tuple[int, int] = 1,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    lower_symbnd_coeffs = lower_symbnd_coeffs.flatten(1)
    upper_symbnd_coeffs = upper_symbnd_coeffs.flatten(1)
    lower_rlx_quad_coeffs = lower_rlx_quad_coeffs.unsqueeze(0)
    lower_rlx_linear_coeffs = lower_rlx_linear_coeffs.unsqueeze(0)
    upper_rlx_quad_coeffs = upper_rlx_quad_coeffs.unsqueeze(0)
    upper_rlx_linear_coeffs = upper_rlx_linear_coeffs.unsqueeze(0)

    (
        new_lower_symbnd_quad_coeffs,
        new_lower_symbnd_linear_coeffs,
        new_lower_symbnd_const,
    ) = _quad_backsub_maxpool2d_after_relu(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_quad_coeffs,
        lower_rlx_linear_coeffs,
        lower_rlx_const,
        upper_rlx_quad_coeffs,
        upper_rlx_linear_coeffs,
        upper_rlx_const,
        input_size=input_size,
        output_size=output_size,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )
    (
        new_upper_symbnd_quad_coeffs,
        new_upper_symbnd_linear_coeffs,
        new_upper_symbnd_const,
    ) = _quad_backsub_maxpool2d_after_relu(
        upper_symbnd_coeffs,
        upper_symbnd_const,
        upper_rlx_quad_coeffs,
        upper_rlx_linear_coeffs,
        upper_rlx_const,
        lower_rlx_quad_coeffs,
        lower_rlx_linear_coeffs,
        lower_rlx_const,
        input_size=input_size,
        output_size=output_size,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )

    return (
        new_lower_symbnd_quad_coeffs,
        new_lower_symbnd_linear_coeffs,
        new_lower_symbnd_const,
        new_upper_symbnd_quad_coeffs,
        new_upper_symbnd_linear_coeffs,
        new_upper_symbnd_const,
    )


def quad_backsub_maxpool2d_after_relu_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_quad_coeffs: Tensor,
    lower_rlx_linear_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_quad_coeffs: Tensor,
    upper_rlx_linear_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    input_size: tuple[int, int, int],
    output_size: tuple[int, int, int],
    kernel_size: tuple[int, int],
    dilation: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    stride: int | tuple[int, int] = 1,
) -> tuple[Tensor, Tensor, Tensor]:
    lower_symbnd_coeffs = lower_symbnd_coeffs.flatten(1)
    lower_rlx_quad_coeffs = lower_rlx_quad_coeffs.unsqueeze(0)
    lower_rlx_linear_coeffs = lower_rlx_linear_coeffs.unsqueeze(0)
    upper_rlx_quad_coeffs = upper_rlx_quad_coeffs.unsqueeze(0)
    upper_rlx_linear_coeffs = upper_rlx_linear_coeffs.unsqueeze(0)

    return _quad_backsub_maxpool2d_after_relu(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_quad_coeffs,
        lower_rlx_linear_coeffs,
        lower_rlx_const,
        upper_rlx_quad_coeffs,
        upper_rlx_linear_coeffs,
        upper_rlx_const,
        input_size=input_size,
        output_size=output_size,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )
