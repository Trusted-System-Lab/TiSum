__all__ = ["quad_relax_sigmoid", "quad_relax_tanh"]

import torch
from torch import Tensor

from ..linear import linear_relax_sigmoid, linear_relax_tanh
from ..._enums import RelaxMode
from ...functions import *

_TANH_LIMIT = 4.5
_SIGMOID_LIMIT = 8


def _cal_quad_coeff_a(l: Tensor, u: Tensor, func: str, is_max: bool = True) -> Tensor:
    """We need calculate the bounds of ddsigmoid given the input domain."""
    if not is_max:
        return -_cal_quad_coeff_a(-u, -l, func, is_max=True)

    if func == "sigmoid":
        right_point = 1.3173183203
        left_point = -right_point
        max_value = 0.0962250531
    elif func == "tanh":
        right_point = 0.6585746408
        left_point = -right_point
        max_value = 0.7698004246
    else:
        raise RuntimeError(f"Not supported function {func}.")

    a = torch.zeros_like(l)

    mask1 = l < left_point
    mask2 = mask1 & (u < left_point)
    if torch.any(mask2):
        a = torch.where(mask2, ddsigmoid(u), a)
    mask2 = mask1 & (u >= left_point)
    if torch.any(mask2):
        a = torch.where(mask2, max_value, a)

    mask1 = (l >= left_point) & (l <= 0)
    if torch.any(mask1):
        a = torch.where(mask1, ddsigmoid(l), a)

    mask1 = (l > 0) & (l <= right_point)
    if torch.any(mask1):
        a = torch.where(mask1, torch.maximum(ddsigmoid(l), ddsigmoid(u)), a)

    mask1 = l > right_point
    if torch.any(mask1):
        a = torch.where(mask1, ddsigmoid(u), a)

    """
    NOTE THAT we use the lower bound for the upper bound and vice versa.
    """
    a.mul_(0.5)  # Theorem: a = (1/2) f''(m)
    return a


def _cal_quad_coeffs_sshape_lower(
    x_l: Tensor,
    x_u: Tensor,
    y_l: Tensor,
    y_u: Tensor,
    add_u_l: Tensor,
    k_lu: Tensor,
    func: str,
) -> tuple[Tensor, Tensor, Tensor]:
    a = _cal_quad_coeff_a(x_l, x_u, func, is_max=True)
    b = k_lu - a * add_u_l
    c = y_l - x_l * (a * x_l + b)

    return a, b, c


def _cal_quad_coeffs_sshape_upper(
    x_l: Tensor,
    x_u: Tensor,
    y_l: Tensor,
    y_u: Tensor,
    add_u_l: Tensor,
    k_lu: Tensor,
    func: str,
) -> tuple[Tensor, Tensor, Tensor]:
    a = _cal_quad_coeff_a(x_l, x_u, func, is_max=False)
    b = k_lu - a * add_u_l
    c = y_u - x_u * (a * x_u + b)

    return a, b, c


"""
NOTE THAT This is an old code that does not provide benefits.
"""


def _cal_quad_coeffs_sshape_left(
    x_l: Tensor,
    x_u: Tensor,
    y_l: Tensor,
    y_u: Tensor,
    add_u_l: Tensor,
    sub_u_l: Tensor,
    k_lu: Tensor,
    dy_l: Tensor,
    dy_u: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    a = (k_lu - dy_l) / sub_u_l
    b = k_lu - a * add_u_l
    c = y_l - x_l * (a * x_l + b)

    return a, b, c


def _cal_quad_coeffs_sshape_left2(
    x_l: Tensor,
    x_u: Tensor,
    y_l: Tensor,
    y_u: Tensor,
    add_u_l: Tensor,
    sub_u_l: Tensor,
    k_lu: Tensor,
    dy_l: Tensor,
    dy_u: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    a = (k_lu - 0) / sub_u_l
    b = k_lu - a * add_u_l
    c = y_l - x_l * (a * x_l + b)

    return a, b, c


def _cal_quad_coeffs_sshape_right(
    x_l: Tensor,
    x_u: Tensor,
    y_l: Tensor,
    y_u: Tensor,
    add_u_l: Tensor,
    sub_u_l: Tensor,
    k_lu: Tensor,
    dy_l: Tensor,
    dy_u: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    a = -(k_lu - dy_u) / sub_u_l
    b = k_lu - a * add_u_l
    c = y_u - x_u * (a * x_u + b)

    return a, b, c


def _cal_quad_coeffs_sshape_right2(
    x_l: Tensor,
    x_u: Tensor,
    y_l: Tensor,
    y_u: Tensor,
    add_u_l: Tensor,
    sub_u_l: Tensor,
    k_lu: Tensor,
    dy_l: Tensor,
    dy_u: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    a = -(k_lu - 0) / sub_u_l
    b = k_lu - a * add_u_l
    c = y_u - x_u * (a * x_u + b)

    return a, b, c


def _quad_relax_sshape(
    l: Tensor,
    u: Tensor,
    func: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

    ori_shape = l.shape
    l = l.flatten()
    u = u.flatten()

    if func == "sigmoid":
        f, df, range_limit = sigmoid, dsigmoid, _SIGMOID_LIMIT
        linear_coeffs_l, const_l, linear_coeffs_u, const_u = linear_relax_sigmoid(
            l, u, mode=RelaxMode.SINGLE_LINEAR
        )
    elif func == "tanh":
        f, df, range_limit = tanh, dtanh, _TANH_LIMIT
        linear_coeffs_l, const_l, linear_coeffs_u, const_u = linear_relax_tanh(
            l, u, mode=RelaxMode.SINGLE_LINEAR
        )
    else:
        raise RuntimeError(f"Not supported function {func}.")

    n = l.numel()
    zeros = torch.zeros(n, dtype=l.dtype, device=l.device)
    quad_coeffs_l = zeros.clone()
    quad_coeffs_u = zeros.clone()

    x_l, x_u, y_l, y_u = l, u, f(l), f(u)
    dy_l, dy_u = df(x_l), df(x_u)
    add_u_l, sub_u_l = x_u + x_l, x_u - x_l
    k_lu = torch.where(x_u != x_l, (y_u - y_l) / sub_u_l, dy_l)

    mask = l >= 0 & (u <= range_limit) & (u - l > 0.1)
    quad_coeffs_l[mask], linear_coeffs_l[mask], const_l[mask] = (
        _cal_quad_coeffs_sshape_lower(
            x_l[mask], x_u[mask], y_l[mask], y_u[mask], add_u_l[mask], k_lu[mask], func
        )
    )

    mask = u <= 0 & (l >= -range_limit) & (u - l > 0.1)
    quad_coeffs_u[mask], linear_coeffs_u[mask], const_u[mask] = (
        _cal_quad_coeffs_sshape_upper(
            x_l[mask], x_u[mask], y_l[mask], y_u[mask], add_u_l[mask], k_lu[mask], func
        )
    )



    return (
        quad_coeffs_l.reshape(ori_shape).detach(),
        linear_coeffs_l.reshape(ori_shape).detach(),
        const_l.reshape(ori_shape).detach(),
        quad_coeffs_u.reshape(ori_shape).detach(),
        linear_coeffs_u.reshape(ori_shape).detach(),
        const_u.reshape(ori_shape).detach(),
    )


def quad_relax_sigmoid(
    pre_lower_bound: Tensor,
    pre_upper_bound: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _quad_relax_sshape(pre_lower_bound, pre_upper_bound, "sigmoid")


def quad_relax_tanh(
    pre_lower_bound: Tensor,
    pre_upper_bound: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return _quad_relax_sshape(pre_lower_bound, pre_upper_bound, "tanh")
