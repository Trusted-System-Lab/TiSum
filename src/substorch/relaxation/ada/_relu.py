__all__ = ["ada_relax_relu", "ada_relax_relu_deepzono"]

import torch
from torch import Tensor

from ..linear import linear_relax_relu, linear_relax_relu_deepzono


def ada_relax_relu(
    x_l: Tensor,
    x_u: Tensor,
    *,
    ada_mask: Tensor,
) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor], Tensor, Tensor | None]:
    s_l, t_l, s_u, t_u = linear_relax_relu(x_l, x_u)

    ada_idx = torch.nonzero(ada_mask).squeeze().detach()
    if ada_idx.dim() == 1:
        ada_idx = ada_idx.unsqueeze(0)

    ada_param = None
    if ada_idx.numel() > 0:
        ada_param = s_l[ada_mask].clone().detach().requires_grad_(True)

    return (s_l, t_l, s_u, t_u), ada_idx, ada_param


def ada_relax_relu_deepzono(
    x_l: Tensor,
    x_u: Tensor,
    *,
    ada_mask: Tensor,
) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor], Tensor, Tensor | None]:
    s_l, t_l, s_u, t_u = linear_relax_relu_deepzono(x_l, x_u)

    ada_idx = torch.nonzero(ada_mask).squeeze().detach()
    if ada_idx.dim() == 1:
        ada_idx = ada_idx.unsqueeze(0)

    ada_param = None
    if ada_idx.numel() > 0:
        ada_param = s_l[ada_mask].clone().detach().requires_grad_(True)

    return (s_l, t_l, s_u, t_u), ada_idx, ada_param
