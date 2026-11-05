"""This module provides PGD (projected gradient descent) algorithms to project a 1D or nD vector onto the 1D or nD simplex."""

__all__ = ["pgd_1d", "pgd_nd", "pgd_nd_duchi"]

import torch
from torch import Tensor


def pgd_1d(
    v: Tensor,
) -> Tensor:
    """Project a vector v onto the 1D simplex."""
    return v.clamp(min=0.0, max=1.0)


def _pgd_nd_duchi(
    v: Tensor,
) -> Tensor:

    m, n = v.size()
    n += 1
    v1 = 1 - v.sum(dim=1, keepdim=True)
    v = torch.hstack((v1, v))

    u = v.sort(dim=1, descending=True)[0]

    torch.use_deterministic_algorithms(False)
    cssv = u.cumsum(dim=1) - 1
    torch.use_deterministic_algorithms(True)

    rho = torch.arange(1, n + 1, device=v.device).float().view(1, -1)
    cond = u - cssv / rho > 0
    k = cond.sum(dim=1) - 1
    theta = cssv[torch.arange(m), k] / (k + 1).float()
    new_v = (v - theta.view(-1, 1)).clamp(min=0.0)

    tol = 1e-4
    v_sum = new_v.sum(dim=1) + tol
    new_v = new_v / v_sum.unsqueeze(1)

    new_v = new_v[:, 1:]

    return new_v




def pgd_nd_duchi(
    v: Tensor,
) -> Tensor:
    """Project each vector in a 2D tensor v onto the nD simplex."""
    return _pgd_nd_duchi(v)


def _pgd_nd(
    v: Tensor,
) -> Tensor:
    tol = 1e-4

    new_v = v.clamp_min(0)
    v_sum = new_v.sum(dim=1) + tol
    v_sum[v_sum == 0] = 1.0

    new_v = new_v / v_sum.unsqueeze(1)

    return new_v


_pgd_nd = torch.jit.trace(_pgd_nd, torch.rand(5, 5) * 2 - 1)


def pgd_nd(
    v: Tensor,
) -> Tensor:
    """Project each vector in a 2D tensor v onto the nD simplex."""
    return _pgd_nd_duchi(v)
