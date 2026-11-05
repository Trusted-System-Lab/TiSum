__all__ = ["quad_backsub_gemm", "quad_backsub_gemm_lower"]

import torch
from torch import Tensor

from ._indices import cal_prod_weight_bias, cal_kwargs_data_of_quad_relaxation_gemm
from ._quad_to_linear import relax_xixi, relax_xixj


def _quad_backsub_gemm(
    Q: Tensor,
    A: Tensor,
    b: Tensor,
    *,
    weight: Tensor,
    bias: Tensor,
    weight_bias_prod: Tensor,
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
    mul_lx_ly: Tensor,
    mul_ux_uy: Tensor,
    mul_lx_uy: Tensor,
    mul_ux_ly: Tensor,
    add_lx_ux: Tensor,
    add_ly_uy: Tensor,
    is_upper: bool,
) -> tuple[Tensor, Tensor]:
    A_quad, b_quad = _relax_quad_to_linear_gemm(
        Q,
        weight_bias_prod=weight_bias_prod,
        l=l,
        u=u,
        h=h,
        sqr_l=sqr_l,
        sqr_u=sqr_u,
        sqr_h=sqr_h,
        k_lu=k_lu,
        k_h=k_h,
        lx=lx,
        ux=ux,
        ly=ly,
        uy=uy,
        mul_lx_ly=mul_lx_ly,
        mul_ux_uy=mul_ux_uy,
        mul_lx_uy=mul_lx_uy,
        mul_ux_ly=mul_ux_ly,
        add_lx_ux=add_lx_ux,
        add_ly_uy=add_ly_uy,
        is_upper=is_upper,
    )

    A_linear = A @ weight
    b_linear = A @ bias

    A_quad.add_(A_linear)
    b_quad.add_(b_linear + b)

    return A_quad, b_quad


def _relax_quad_to_linear_gemm(
    Q: Tensor,
    *,
    weight_bias_prod: Tensor,
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
    mul_lx_ly: Tensor,
    mul_ux_uy: Tensor,
    mul_lx_uy: Tensor,
    mul_ux_ly: Tensor,
    add_lx_ux: Tensor,
    add_ly_uy: Tensor,
    is_upper: bool,
) -> tuple[Tensor, Tensor]:
    Q = _expand_quad(Q, weight_bias_prod.to(Q.device))
    Q, A_quad_linear, b_quad_linear = _divide_quad(Q)
    A_quad, b_quad = _relax_quad_to_linear(
        Q,
        l=l,
        u=u,
        h=h,
        sqr_l=sqr_l,
        sqr_u=sqr_u,
        sqr_h=sqr_h,
        k_lu=k_lu,
        k_h=k_h,
        lx=lx,
        ux=ux,
        ly=ly,
        uy=uy,
        mul_lx_ly=mul_lx_ly,
        mul_ux_uy=mul_ux_uy,
        mul_lx_uy=mul_lx_uy,
        mul_ux_ly=mul_ux_ly,
        add_lx_ux=add_lx_ux,
        add_ly_uy=add_ly_uy,
        is_upper=is_upper,
    )




    A = A_quad + A_quad_linear
    b = b_quad + b_quad_linear

    return A, b


def _expand_quad(
    Q: Tensor,
    weight_bias_prod: Tensor,
) -> Tensor:
    Q = torch.einsum("bm, mij -> bij", Q, weight_bias_prod)

    return Q


def _divide_quad(
    Q: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    b = Q[:, -1, -1]
    A = Q[:, :-1, -1] * 2
    Q = Q[:, :-1, :-1]

    return Q, A, b


def _relax_quad_to_linear(
    Q: Tensor,
    *,
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
    mul_lx_ly: Tensor,
    mul_ux_uy: Tensor,
    mul_lx_uy: Tensor,
    mul_ux_ly: Tensor,
    add_lx_ux: Tensor,
    add_ly_uy: Tensor,
    is_upper: bool,
) -> tuple[Tensor, Tensor]:
    Q_diag = torch.diagonal(Q, dim1=-2, dim2=-1).clone()
    A_xx, b_xx = relax_xixi(Q_diag, l, sqr_l, h, sqr_h, k_lu, k_h, is_upper)

    Q_nondiag = Q
    Q_nondiag.diagonal(dim1=-2, dim2=-1).fill_(0)
    A_xy, b_xy = relax_xixj(
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

    Q.diagonal(dim1=-2, dim2=-1).copy_(Q_diag)

    A = A_xx.add_(A_xy)
    b = b_xx.add_(b_xy)

    return A, b


def quad_backsub_gemm(
    lower_symbnd_quad_coeffs: Tensor,
    lower_symbnd_linear_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_quad_coeffs: Tensor,
    upper_symbnd_linear_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    pre_lower_bound: Tensor,
    pre_upper_bound: Tensor,
    weight: Tensor,
    bias: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    weight = weight.t()
    weight_bias_prod = cal_prod_weight_bias(weight, bias)

    kwargs_data = cal_kwargs_data_of_quad_relaxation_gemm(
        pre_lower_bound, pre_upper_bound
    )

    new_lower_coeffs, new_lower_const = _quad_backsub_gemm(
        lower_symbnd_quad_coeffs,
        lower_symbnd_linear_coeffs,
        lower_symbnd_const,
        weight=weight,
        bias=bias,
        weight_bias_prod=weight_bias_prod,
        l=pre_lower_bound,
        u=pre_upper_bound,
        **kwargs_data,
        is_upper=False,
    )
    new_upper_coeffs, new_upper_const = _quad_backsub_gemm(
        upper_symbnd_quad_coeffs,
        upper_symbnd_linear_coeffs,
        upper_symbnd_const,
        weight=weight,
        bias=bias,
        weight_bias_prod=weight_bias_prod,
        l=pre_lower_bound,
        u=pre_upper_bound,
        **kwargs_data,
        is_upper=True,
    )
    return new_lower_coeffs, new_lower_const, new_upper_coeffs, new_upper_const


def quad_backsub_gemm_lower(
    lower_symbnd_quad_coeffs: Tensor,
    lower_symbnd_linear_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    pre_lower_bound: Tensor,
    pre_upper_bound: Tensor,
    weight: Tensor,
    bias: Tensor,
) -> tuple[Tensor, Tensor]:
    weight = weight.t()
    weight_bias_prod = cal_prod_weight_bias(weight, bias)

    kwargs_data = cal_kwargs_data_of_quad_relaxation_gemm(
        pre_lower_bound, pre_upper_bound
    )

    new_lower_coeffs, new_lower_const = _quad_backsub_gemm(
        lower_symbnd_quad_coeffs,
        lower_symbnd_linear_coeffs,
        lower_symbnd_const,
        weight=weight,
        bias=bias,
        weight_bias_prod=weight_bias_prod,
        l=pre_lower_bound,
        u=pre_upper_bound,
        **kwargs_data,
        is_upper=False,
    )

    return new_lower_coeffs, new_lower_const
