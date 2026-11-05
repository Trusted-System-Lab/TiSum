"""This module provides PGD (projected gradient descent) algorithms to project a 1D or nD vector onto the 1D or nD simplex."""

__all__ = ["pgd_nd_duchi"]

import torch
from torch import Tensor


def _pgd_nd_duchi(
    v: Tensor,
    is_sum_one: bool = True,
) -> Tensor:

    m, n = v.size()
    if not is_sum_one:
        n += 1
        v1 = 1 - v.sum(dim=1, keepdim=True)
        v = torch.hstack((v1, v))
    else:
        v[:, 0] = 1 - v.sum(dim=1)

    u = v.sort(dim=1, descending=True)[0]

    torch.use_deterministic_algorithms(False)
    cssv = u.cumsum(dim=1) - 1
    torch.use_deterministic_algorithms(True)

    rho = torch.arange(1, n + 1, device=v.device).float().view(1, -1)
    cond = u - cssv / rho > 0
    k = cond.sum(dim=1) - 1
    theta = cssv[torch.arange(m), k] / (k + 1).float()
    new_v = (v - theta.view(-1, 1)).clamp_min(0)

    tol = 1e-4
    v_sum = new_v.sum(dim=1) + tol
    new_v = new_v / v_sum.unsqueeze(1)

    assert torch.all(new_v.sum(dim=1) <= 1.0), f"{new_v.sum(dim=1)}"
    assert (new_v >= 0).all(), f"{new_v}"
    assert (new_v <= 1).all(), f"{new_v}"

    if not is_sum_one:
        new_v = new_v[:, 1:]

    return new_v




def pgd_nd_duchi(
    v: Tensor,
    is_sum_one: bool = False,
) -> Tensor:
    """Project each vector in a 2D tensor v onto the nD simplex."""
    return _pgd_nd_duchi(v, is_sum_one=is_sum_one)



