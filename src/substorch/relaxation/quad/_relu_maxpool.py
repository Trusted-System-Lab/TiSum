__all__ = ["quad_relax_maxpool2d_after_relu"]

import torch.nn.functional as F
from torch import Tensor

from ...relax_mask import get_rlx_mask_maxpool2d
from ...utils import cal_l_argmax_maxpool2d, unfold_pre_bound_maxpool2d

U_MAX_USE_QUAD = 0.1


def quad_relax_maxpool2d_after_relu(
    x_l: Tensor,
    x_u: Tensor,
    *,
    kernel_size: tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    dilation: int | tuple[int, int],
    lower_relu_quad_coeffs: Tensor,
    lower_relu_linear_coeffs: Tensor,
    lower_relu_const: Tensor,
    upper_relu_quad_coeffs: Tensor,
    upper_relu_linear_coeffs: Tensor,
    upper_relu_const: Tensor,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:
    unfold_l, unfold_u = unfold_pre_bound_maxpool2d(
        x_l,
        x_u,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    l_max, l_argmax = cal_l_argmax_maxpool2d(unfold_l, unfold_u)

    l, u = unfold_l, unfold_u
    c, h1, w1 = lower_relu_quad_coeffs.shape
    kk = kernel_size[0] * kernel_size[1]

    def unfold_reshape(t: Tensor) -> Tensor:
        args = (kernel_size, dilation, padding, stride)
        return (
            F.unfold(t.unsqueeze(1), *args)
            .squeeze()
            .reshape(c, kk, -1)
            .permute(0, 2, 1)
            .reshape(-1, kk)
        )

    pool_quad_coeffs_l = unfold_reshape(lower_relu_quad_coeffs)
    pool_quad_coeffs_u = unfold_reshape(upper_relu_quad_coeffs)
    pool_linear_coeffs_l = unfold_reshape(lower_relu_linear_coeffs)
    pool_linear_coeffs_u = unfold_reshape(upper_relu_linear_coeffs)
    pool_const_l = unfold_reshape(lower_relu_const)
    pool_const_u = unfold_reshape(upper_relu_const)

    """
    The lower bound we use the mean weighted sum of all the lower bounds of the inputs.
    The upper bound we use the sum of the upper bounds of the inputs.

    For upper bound:
    Note that each preceding ReLU neuron only considers one input variable in the
    pool. For quadratic and linear terms, we need not sum them up because they are
    represented by coefficients symmetrically. Hence, the unfolded inputs have
    provided us with the correct coefficients.
    """

    mask_nontriv = get_rlx_mask_maxpool2d(unfold_l, unfold_u, l_max, l_argmax)
    mask_zero = (l == 0).all(dim=1) & (u == 0).all(dim=1)
    mask_triv = ~(mask_zero | mask_nontriv)

    pool_quad_coeffs_l[mask_zero] = 0.0
    pool_linear_coeffs_l[mask_zero] = 0.0
    pool_const_l[mask_zero] = 0.0
    pool_quad_coeffs_u[mask_zero] = 0.0
    pool_linear_coeffs_u[mask_zero] = 0.0
    pool_const_u[mask_zero] = 0.0

    temp = pool_quad_coeffs_l[mask_triv, l_argmax[mask_triv]]
    pool_quad_coeffs_l[mask_triv] = 0.0
    pool_quad_coeffs_l[mask_triv, l_argmax[mask_triv]] = temp
    temp = pool_linear_coeffs_l[mask_triv, l_argmax[mask_triv]]
    pool_linear_coeffs_l[mask_triv] = 0.0
    pool_linear_coeffs_l[mask_triv, l_argmax[mask_triv]] = temp
    temp = pool_const_l[mask_triv, l_argmax[mask_triv]]
    pool_const_l[mask_triv] = 0.0
    pool_const_l[mask_triv, l_argmax[mask_triv]] = temp
    temp = pool_quad_coeffs_u[mask_triv, l_argmax[mask_triv]]
    pool_quad_coeffs_u[mask_triv] = 0.0
    pool_quad_coeffs_u[mask_triv, l_argmax[mask_triv]] = temp
    temp = pool_linear_coeffs_u[mask_triv, l_argmax[mask_triv]]
    pool_linear_coeffs_u[mask_triv] = 0.0
    pool_linear_coeffs_u[mask_triv, l_argmax[mask_triv]] = temp
    temp = pool_const_u[mask_triv, l_argmax[mask_triv]]
    pool_const_u[mask_triv] = 0.0
    pool_const_u[mask_triv, l_argmax[mask_triv]] = temp

    pool_const_l = pool_const_l.sum(dim=1)
    pool_const_u = pool_const_u.sum(dim=1)

    pool_quad_coeffs_l[mask_nontriv] = pool_quad_coeffs_l[mask_nontriv] / kk
    pool_linear_coeffs_l[mask_nontriv] = pool_linear_coeffs_l[mask_nontriv] / kk
    pool_const_l[mask_nontriv] = pool_const_l[mask_nontriv] / kk

    return (
        pool_quad_coeffs_l,
        pool_linear_coeffs_l,
        pool_const_l,
        pool_quad_coeffs_u,
        pool_linear_coeffs_u,
        pool_const_u,
    )
