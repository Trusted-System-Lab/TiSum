__all__ = ["linear_relax_maxpool2d"]

import torch
from torch import Tensor

from ...relax_mask import get_rlx_mask_maxpool2d
from ...utils import cal_l_argmax_maxpool2d, unfold_pre_bound_maxpool2d


def _upper_bound_maxpool2d(
    x_u: Tensor,
    t_u: Tensor,
    nontriv_mask: Tensor,
) -> Tensor:
    """Before this function, we have already check the trivial cases: The MaxPool always outputs one of the inputs."""

    t_u[nontriv_mask] = x_u.max(dim=1)[0][nontriv_mask]

    return t_u


def _upper_bound_maxpool2d_ehlers(
    x_l: Tensor,
    max_x_l: Tensor,
    s_u: Tensor,
    t_u: Tensor,
    nontriv_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Here we use an upper bound proposed by Ehlers in the paper: https://arxiv.org/pdf/1705.01320 The practical results demonstrate that this relaxation is not as tight as the one used in _cal_rlx_maxpool2d."""
    s_u[nontriv_mask] = 1.0
    t_u[nontriv_mask] = max_x_l[nontriv_mask] - x_l.sum(dim=1)[nontriv_mask]

    return s_u, t_u


def linear_relax_maxpool2d(
    x_l: Tensor,
    x_u: Tensor,
    *,
    kernel_size: tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """NOTE: This is only for a MaxPool layer after a ReLU layer."""
    assert torch.all(x_l >= 0)
    assert torch.all(x_u >= 0)

    dtype, device = x_l.dtype, x_l.device

    x_l, x_u = unfold_pre_bound_maxpool2d(
        x_l,
        x_u,
        kernel_size=kernel_size,
        padding=padding,
        dilation=dilation,
        stride=stride,
    )
    n, s = x_l.shape
    max_x_l, argmax_x_l = cal_l_argmax_maxpool2d(x_l, x_u)
    mask_nontriv = get_rlx_mask_maxpool2d(x_l, x_u, max_x_l, argmax_x_l)

    """
    Before this function, we have already check the trivial cases:
    The MaxPool always outputs one of the inputs.

    But we still need to consider the following cases here:
    Degenerated cases:
    Given a set of intervals as the inputs of the MaxPool function, how to filter out 
    which intervals are never the outputs? 
    Check each interval by if its upper bound is less than the lower bound of all the 
    rest intervals. If so, this interval is never the output.

    BUT HERE THE RELAXATION DOES NOT CARE ABOUT THE DEGENERATED CASES.
    """
    s_l = torch.zeros((n, s), dtype=dtype, device=device)
    t_l = torch.zeros((n,), dtype=dtype, device=device)
    s_u = s_l.clone()
    t_u = t_l.clone()
    idx = torch.arange(n, device=device)

    """
    This is only for ReLU. For other activation functions, it seems that such cases
    will not happen. That is all inputs are constants instead of intervals.
    """
    mask_z = (x_l == 0).all(dim=1) & (x_u == 0).all(dim=1)

    mask_l = ~mask_z
    idx_masked = idx[mask_l]
    l_argmax_masked = argmax_x_l[mask_l]
    s_l[idx_masked, l_argmax_masked] = 1.0

    mask_t = ~(mask_nontriv | mask_z)
    s_u[mask_t, argmax_x_l[mask_t]] = 1.0

    t_u = _upper_bound_maxpool2d(x_u, t_u, mask_nontriv)

    return s_l.detach_(), t_l.detach_(), s_u.detach_(), t_u.detach_()
