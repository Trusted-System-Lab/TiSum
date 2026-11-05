__all__ = ["ada_relax_maxpool2d"]

import torch
from torch import Tensor

from ..linear import linear_relax_maxpool2d


def ada_relax_maxpool2d(
    x_l: Tensor,
    x_u: Tensor,
    *,
    ada_mask: Tensor,
    kernel_size: tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor], Tensor, Tensor | None]:
    s_l, s_u, t_l, t_u = linear_relax_maxpool2d(
        x_l,
        x_u,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    ada_idx = torch.nonzero(ada_mask).squeeze().detach()
    ada_param = None
    if ada_idx.numel() > 0:
        ada_param = s_l[ada_idx].clone().detach().requires_grad_(True)

    return (s_l, t_l, s_u, t_u), ada_idx, ada_param
