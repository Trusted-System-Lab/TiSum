__all__ = ["linear_relax_relu", "linear_relax_relu_deepzono"]

import torch
from torch import Tensor


def linear_relax_relu_deepzono(
    x_l: Tensor,
    x_u: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute the linear relaxation of ReLU using the DeepZono method."""
    s_u = torch.where(x_l >= 0, 1.0, 0.0)

    mask_u = (x_l < 0) & (x_u > 0)
    s_u = torch.where(mask_u, x_u / (x_u - x_l), s_u)
    t_u = torch.where(mask_u, -x_l * s_u, 0)

    s_l = s_u.clone()
    t_l = torch.zeros_like(x_l)

    return s_l.detach_(), t_l.detach_(), s_u.detach_(), t_u.detach_()


def linear_relax_relu(
    x_l: Tensor,
    x_u: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute the linear relaxation of ReLU using the DeepPoly/CROWN method."""
    s_u = torch.where(x_l >= 0, 1.0, 0.0)

    mask_u = (x_l < 0) & (x_u > 0)
    s_u = torch.where(mask_u, x_u / (x_u - x_l), s_u)
    t_u = torch.where(mask_u, -x_l * s_u, 0)

    s_l = torch.where(x_u > -x_l, 1.0, 0.0)
    t_l = torch.zeros_like(x_l)

    return s_l.detach_(), t_l.detach_(), s_u.detach_(), t_u.detach_()
