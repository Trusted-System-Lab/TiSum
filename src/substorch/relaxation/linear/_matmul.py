__all__ = ["linear_relax_matmul"]

import torch
from torch import Tensor


def linear_relax_matmul(
    xa_l: Tensor,
    xa_u: Tensor,
    xb_l: Tensor,
    xb_u: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """The lower bound is xy <= xb_l * x + xa_l * y - xa_l * xb_l xy <= xb_u * x + xa_u * y - xa_u * xb_u The upper bound is xy >= xb_l * x + xa_u * y - xa_u * xb_l xy >= xb_u * x + xa_l * y - xa_l * xb_u."""
    assert torch.all(xa_l < xa_u)
    assert torch.all(xb_l < xb_u)

    m, n = xa_l.shape
    p = xb_l.shape[1]

    sa_l1 = xb_l.repeat((m, 1, 1))
    sa_l2 = xb_u.repeat((m, 1, 1))
    sb_l1 = xa_l.repeat((1, 1, p))
    sb_l2 = xa_u.repeat((1, 1, p))
    t_l1 = -xa_l.unsqueeze(-1) * xb_l.unsqueeze(0)
    t_l2 = -xa_u.unsqueeze(-1) * xb_u.unsqueeze(0)
    mask = t_l1 > t_l2
    sa_l = torch.where(mask, sa_l1, sa_l2)
    sb_l = torch.where(mask, sb_l1, sb_l2)
    t_l = torch.where(mask, t_l1, t_l2).sum(dim=1)

    sa_u1 = sa_l1
    sa_u2 = sa_l2
    sb_u1 = sb_l2
    sb_u2 = sb_l1
    t_u1 = xa_u.unsqueeze(-1) * xb_l.unsqueeze(0)
    t_u2 = xa_l.unsqueeze(-1) * xb_u.unsqueeze(0)
    mask = t_u1 < t_u2
    sa_u = torch.where(mask, sa_u1, sa_u2)
    sb_u = torch.where(mask, sb_u1, sb_u2)
    t_u = torch.where(mask, t_u1, t_u2).sum(dim=1)

    return (
        sa_l.detach_(),
        sb_l.detach_(),
        t_l.detach_(),
        sa_u.detach_(),
        sb_u.detach_(),
        t_u.detach_(),
    )
