__all__ = ["relax_xixi_one_batch", "relax_xixj_one_batch", "relax_xixi", "relax_xixj"]

import torch
from torch import Tensor

"""
Calculate the parabola with upward opening.

Upper bound is the line crossing (l, l^2) and (u, u^2)
Lower bound is the tangent line at (l + u) / 2
"""


def relax_xixi_one_batch(
    Q_diag: Tensor,
    l: Tensor,
    sqr_l: Tensor,
    h: Tensor,
    sqr_h: Tensor,
    slope_lu: Tensor,
    slope_h: Tensor,
    is_upper: bool,
) -> tuple[Tensor, Tensor]:
    xixi_coeffs = Q_diag

    mask = xixi_coeffs > 0 if is_upper else xixi_coeffs < 0

    A = xixi_coeffs * torch.where(mask, slope_lu, slope_h)
    b = (
        (-A * torch.where(mask, l, h)).add_(
            xixi_coeffs * torch.where(mask, sqr_l, sqr_h)
        )
    ).sum()

    return A, b


"""
The lower bounds have two:
    l_y * x + l_x * y - l_x * l_y and u_y * x + u_x * y - u_x * u_y
The upper bounds have two:
    u_y * x + l_x * y - l_x * u_y and l_y * x + u_x * y - u_x * l_y

The coefficient of x is (l_y + u_y + l_x + u_x) * quad_coeffs_l
The coefficient of y is (l_y + u_y + l_x + u_x) * quad_coeffs_u
The constant term is (- l_x * l_y - u_x * u_y) / 2 * quad_coeffs_l
The constant term is (- l_x * u_y - u_x * l_y) / 2 * quad_coeffs_u
"""


def relax_xixj_one_batch(
    Q_nondiag: Tensor,
    lx: Tensor,
    ly: Tensor,
    ux: Tensor,
    uy: Tensor,
    mul_lx_ly: Tensor,
    mul_ux_uy: Tensor,
    mul_lx_uy: Tensor,
    mul_ux_ly: Tensor,
    add_lx_ux: Tensor,
    add_ly_uy: Tensor,
    is_upper: bool,
) -> tuple[Tensor, Tensor]:
    xixj_coeffs = Q_nondiag

    mask = xixj_coeffs > 0 if is_upper else xixj_coeffs < 0

    xixj_coeffs = 2 * torch.triu(xixj_coeffs, diagonal=1)

    b1 = torch.sum(torch.where(mask, -mul_lx_uy, -mul_lx_ly) * xixj_coeffs)
    b2 = torch.sum(torch.where(mask, -mul_ux_ly, -mul_ux_uy) * xixj_coeffs)
    A1 = (
        (lx * xixj_coeffs)
        .sum(dim=1)
        .add_((torch.where(mask, uy, ly) * xixj_coeffs).sum(dim=0))
    )
    A2 = (
        (ux * xixj_coeffs)
        .sum(dim=1)
        .add_((torch.where(mask, ly, uy) * xixj_coeffs).sum(dim=0))
    )


    mask = b1 > b2 if not is_upper else b1 < b2
    b = torch.where(mask, b1, b2)
    A = torch.where(mask.unsqueeze(-1), A1, A2)


    return A, b


relax_xixi = torch.vmap(
    relax_xixi_one_batch,
    in_dims=(0, *([None] * 7)),
    out_dims=(0, 0),
)

relax_xixj = torch.vmap(
    relax_xixj_one_batch,
    in_dims=(0, *([None] * 11)),
    out_dims=(0, 0),
    randomness="different",
)
