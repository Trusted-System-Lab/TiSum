__all__ = ["quad_relax_relu"]

import torch
from torch import Tensor


def quad_relax_relu(
    x_l: Tensor,
    x_u: Tensor,
    use_lower_quad_rlx: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    ori_shape = x_l.shape
    l, u = x_l.flatten(), x_u.flatten()

    n = l.numel()
    zeros = torch.zeros(n, dtype=l.dtype, device=l.device)
    quad_coeffs_l = zeros.clone()
    quad_coeffs_u = zeros.clone()
    const_l = zeros.clone()
    const_u = zeros.clone()

    mask = l >= 0
    linear_coeffs_u = torch.where(mask, 1.0, 0.0)
    linear_coeffs_l = torch.where(mask, 1.0, 0.0)

    mask0 = u > -l
    mask = (l < 0) & (u > 0)
    mask2 = mask & mask0


    linear_coeffs_l = torch.where(mask2, 1.0, linear_coeffs_l)

    mask1 = mask & ~mask0
    mask2 = mask & mask0

    rcp_sub_u_l = torch.reciprocal(u - l)
    sqr_rcp_sub_u_l = rcp_sub_u_l**2
    mul_u_sqr_rcp_sub_u_l = u * sqr_rcp_sub_u_l
    mul_l_sqr_rcp_sub_u_l = l * sqr_rcp_sub_u_l
    sqr_l = l**2
    sqr_u = u**2

    quad_coeffs_u = torch.where(mask1, mul_u_sqr_rcp_sub_u_l, quad_coeffs_u)
    linear_coeffs_u = torch.where(
        mask1, -2 * l * mul_u_sqr_rcp_sub_u_l, linear_coeffs_u
    )
    const_u = torch.where(mask1, sqr_l * mul_u_sqr_rcp_sub_u_l, const_u)

    quad_coeffs_u = torch.where(mask2, -mul_l_sqr_rcp_sub_u_l, quad_coeffs_u)
    linear_coeffs_u = torch.where(
        mask2, (sqr_l + sqr_u) * sqr_rcp_sub_u_l, linear_coeffs_u
    )
    const_u = torch.where(mask2, -sqr_u * mul_l_sqr_rcp_sub_u_l, const_u)

    """
    NOTE: It seems that the lower bound does not provide benefits.
    This is only used a ReLU before a MaxPool2d layer.
    """
    if use_lower_quad_rlx:
        quad_coeffs_l = torch.where(mask, rcp_sub_u_l, quad_coeffs_l)
        linear_coeffs_l = torch.where(mask, -l * rcp_sub_u_l, linear_coeffs_l)

    return (
        quad_coeffs_l.reshape(ori_shape).detach(),
        linear_coeffs_l.reshape(ori_shape).detach(),
        const_l.reshape(ori_shape).detach(),
        quad_coeffs_u.reshape(ori_shape).detach(),
        linear_coeffs_u.reshape(ori_shape).detach(),
        const_u.reshape(ori_shape).detach(),
    )
