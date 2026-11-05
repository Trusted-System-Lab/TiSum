__all__ = ["quad_backsub_conv2d", "quad_backsub_conv2d_lower"]

import math
import torch
import torch.nn.functional as F
from torch import Tensor

from ._indices import (
    cal_prod_weight_weight,
    cal_mul_weight_2bias,
    cal_square_bias,
    cal_kwargs_data_of_quad_relaxation_conv2d,
)
from ._quad_to_linear import (
    relax_xixi_one_batch,
    relax_xixj_one_batch,
)


def _relax_quad_one_batch_scatter(
    Q: Tensor,
    Q_fold: Tensor,
    weight_prod: Tensor,
    l: Tensor,
    h: Tensor,
    sqr_l: Tensor,
    sqr_h: Tensor,
    k_lu: Tensor,
    k_h: Tensor,
    lx: Tensor,
    ux: Tensor,
    ly: Tensor,
    uy: Tensor,
    add_lx_ux: Tensor,
    add_ly_uy: Tensor,
    mul_lx_ly: Tensor,
    mul_ux_uy: Tensor,
    mul_lx_uy: Tensor,
    mul_ux_ly: Tensor,
    is_upper: bool,
    input_size: tuple[int, int, int],
    output_size: tuple[int, int, int],
    kernel_size: tuple[int, int],
    idx_in: Tensor,
    idx_out: Tensor,
    idx_out2: Tensor,
) -> tuple[Tensor, Tensor]:
    chw_in = math.prod(input_size)
    Q_fold = Q_fold.flatten()

    Q_unfold = torch.einsum("chw, ck -> hwk", Q, weight_prod).flatten()
    Q_fold.scatter_add_(0, idx_in, Q_unfold[idx_out])
    Q_fold = Q_fold.reshape(chw_in, chw_in)

    Q_diag = torch.diagonal(Q_fold, dim1=-2, dim2=-1).clone()
    A_xx, b_xx = relax_xixi_one_batch(Q_diag, l, sqr_l, h, sqr_h, k_lu, k_h, is_upper)

    Q_nondiag = Q_fold
    Q_nondiag.diagonal(dim1=-2, dim2=-1).fill_(0)
    A_xy, b_xy = relax_xixj_one_batch(
        Q_nondiag,
        lx,
        ly,
        ux,
        uy,
        mul_lx_ly,
        mul_ux_uy,
        mul_lx_uy,
        mul_ux_ly,
        add_lx_ux,
        add_ly_uy,
        is_upper,
    )

    Q_fold.diagonal(dim1=-2, dim2=-1).copy_(Q_diag)

    A = A_xx.add_(A_xy)
    b = b_xx.add_(b_xy)

    return A, b


_relax_quad_scatter = torch.vmap(
    _relax_quad_one_batch_scatter,
    in_dims=(0, 0, *([None] * 24)),
    out_dims=(0, 0),
    randomness="different",
)


def _handle_quad_poly_quad_part_scatter(
    Q: Tensor,
    weight_prod: Tensor,
    l: Tensor,
    u: Tensor,
    h: Tensor,
    sqr_l: Tensor,
    sqr_u: Tensor,
    sqr_h: Tensor,
    k_lu: Tensor,
    k_h: Tensor,
    lx: Tensor,
    ux: Tensor,
    ly: Tensor,
    uy: Tensor,
    add_lx_ux: Tensor,
    add_ly_uy: Tensor,
    mul_lx_ly: Tensor,
    mul_ux_uy: Tensor,
    mul_lx_uy: Tensor,
    mul_ux_ly: Tensor,
    idx_in: Tensor,
    idx_out: Tensor,
    idx_out2: Tensor,
    is_upper: bool,
    input_size: tuple[int, int, int],
    output_size: tuple[int, int, int],
    kernel_size: tuple[int, int],
) -> tuple[Tensor, Tensor]:
    l = l.flatten()

    c_in, h_in, w_in = input_size
    chw1 = c_in * h_in * w_in

    Q_fold = torch.zeros((Q.shape[0], chw1, chw1), device=Q.device, dtype=Q.dtype)
    A, b = _relax_quad_scatter(
        Q,
        Q_fold,
        weight_prod,
        l,
        h,
        sqr_l,
        sqr_h,
        k_lu,
        k_h,
        lx,
        ux,
        ly,
        uy,
        add_lx_ux,
        add_ly_uy,
        mul_lx_ly,
        mul_ux_uy,
        mul_lx_uy,
        mul_ux_ly,
        is_upper,
        input_size,
        output_size,
        kernel_size,
        idx_in,
        idx_out,
        idx_out2,
    )

    A = A.reshape(-1, c_in, h_in, w_in)

    return A, b


def _handle_quad_poly_linear_part(
    Q: Tensor,
    weight_mul_2bias: Tensor | None,
    sqr_bias: Tensor | None,
    stride: tuple[int, int],
    padding: tuple[int, int],
    output_padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[Tensor | None, Tensor | None]:
    b = torch.einsum("bchw, c -> b", Q, sqr_bias)

    A = F.conv_transpose2d(
        Q,
        weight_mul_2bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
    )

    return A, b


def _handle_linear_poly(
    A: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    stride: tuple[int, int],
    padding: tuple[int, int],
    output_padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[Tensor, Tensor | None]:
    b = torch.einsum("bchw, c -> b", A, bias)

    A = F.conv_transpose2d(
        A,
        weight,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
    )

    return A, b


def _quad_backsub_conv2d(
    Q: Tensor,
    A: Tensor,
    b: Tensor,
    *,
    weight: Tensor,
    bias: Tensor,
    mul_weight_2bias: Tensor,
    prod_weight_weight: Tensor,
    square_bias: Tensor,
    l: Tensor,
    u: Tensor,
    h: Tensor,
    sqr_l: Tensor,
    sqr_u: Tensor,
    sqr_h: Tensor,
    k_lu: Tensor,
    k_h: Tensor,
    lx: Tensor,
    ux: Tensor,
    ly: Tensor,
    uy: Tensor,
    add_lx_ux: Tensor,
    add_ly_uy: Tensor,
    mul_lx_ly: Tensor,
    mul_ux_uy: Tensor,
    mul_lx_uy: Tensor,
    mul_ux_ly: Tensor,
    idx_in: Tensor,
    idx_out: Tensor,
    idx_out2: Tensor,
    is_upper: bool,
    input_size: tuple[int, int, int],
    output_size: tuple[int, int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    output_padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
) -> tuple[Tensor, Tensor]:
    conv_args = (stride, padding, output_padding, dilation)

    A_linear, b_linear = _handle_linear_poly(A, weight, bias, *conv_args)
    A = A_linear
    b = b + b_linear
    A_quad_linear, b_quad_linear = _handle_quad_poly_linear_part(
        Q, mul_weight_2bias, square_bias, *conv_args
    )
    A.add_(A_quad_linear)
    b.add_(b_quad_linear)

    A_quad, b_quad = _handle_quad_poly_quad_part_scatter(
        Q,
        prod_weight_weight,
        l,
        u,
        h,
        sqr_l,
        sqr_u,
        sqr_h,
        k_lu,
        k_h,
        lx,
        ux,
        ly,
        uy,
        add_lx_ux,
        add_ly_uy,
        mul_lx_ly,
        mul_ux_uy,
        mul_lx_uy,
        mul_ux_ly,
        idx_in,
        idx_out,
        idx_out2,
        is_upper,
        input_size,
        output_size,
        kernel_size,
    )

    A.add_(A_quad)
    b.add_(b_quad)

    return A, b


def quad_backsub_conv2d(
    lower_symbnd_quad_coeffs: Tensor,
    lower_symbnd_linear_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_quad_coeffs: Tensor,
    upper_symbnd_linear_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    weight: Tensor,
    bias: Tensor,
    input_size: tuple[int, int, int],
    output_size: tuple[int, int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    output_padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
    pre_lower_bound: Tensor,
    pre_upper_bound: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    assert groups == 1, "Groups > 1 is not supported in quad backsub conv2d."

    mul_weight_2bias = cal_mul_weight_2bias(weight, bias)
    prod_weight_weight = cal_prod_weight_weight(weight)
    square_bias = cal_square_bias(bias)
    kernel_size = (weight.size(-2), weight.size(-1))
    kwargs_data = cal_kwargs_data_of_quad_relaxation_conv2d(
        pre_lower_bound,
        pre_upper_bound,
        input_size=input_size,
        output_size=output_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    new_lower_symbnd_coeffs, new_lower_symbnd_const = _quad_backsub_conv2d(
        lower_symbnd_quad_coeffs,
        lower_symbnd_linear_coeffs,
        lower_symbnd_const,
        weight=weight,
        bias=bias,
        mul_weight_2bias=mul_weight_2bias,
        prod_weight_weight=prod_weight_weight,
        square_bias=square_bias,
        l=pre_lower_bound,
        u=pre_upper_bound,
        **kwargs_data,
        is_upper=False,
        input_size=input_size,
        output_size=output_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
    )

    new_upper_symbnd_coeffs, new_upper_symbnd_const = _quad_backsub_conv2d(
        upper_symbnd_quad_coeffs,
        upper_symbnd_linear_coeffs,
        upper_symbnd_const,
        weight=weight,
        bias=bias,
        mul_weight_2bias=mul_weight_2bias,
        prod_weight_weight=prod_weight_weight,
        square_bias=square_bias,
        l=pre_lower_bound,
        u=pre_upper_bound,
        **kwargs_data,
        is_upper=True,
        input_size=input_size,
        output_size=output_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
    )

    return (
        new_lower_symbnd_coeffs,
        new_lower_symbnd_const,
        new_upper_symbnd_coeffs,
        new_upper_symbnd_const,
    )


def quad_backsub_conv2d_lower(
    lower_symbnd_quad_coeffs: Tensor,
    lower_symbnd_linear_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    weight: Tensor,
    bias: Tensor,
    input_size: tuple[int, int, int],
    output_size: tuple[int, int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    output_padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
    pre_lower_bound: Tensor,
    pre_upper_bound: Tensor,
) -> tuple[Tensor, Tensor]:
    assert groups == 1, "Groups > 1 is not supported in quad backsub conv2d."

    mul_weight_2bias = cal_mul_weight_2bias(weight, bias)
    prod_weight_weight = cal_prod_weight_weight(weight)
    square_bias = cal_square_bias(bias)
    kernel_size = (weight.size(-2), weight.size(-1))
    kwargs_data = cal_kwargs_data_of_quad_relaxation_conv2d(
        pre_lower_bound,
        pre_upper_bound,
        input_size=input_size,
        output_size=output_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    new_lower_symbnd_coeffs, new_lower_symbnd_const = _quad_backsub_conv2d(
        lower_symbnd_quad_coeffs,
        lower_symbnd_linear_coeffs,
        lower_symbnd_const,
        weight=weight,
        bias=bias,
        mul_weight_2bias=mul_weight_2bias,
        prod_weight_weight=prod_weight_weight,
        square_bias=square_bias,
        l=pre_lower_bound,
        u=pre_upper_bound,
        **kwargs_data,
        is_upper=False,
        input_size=input_size,
        output_size=output_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
    )

    return new_lower_symbnd_coeffs, new_lower_symbnd_const
