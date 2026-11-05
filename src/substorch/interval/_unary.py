__all__ = [
    "interval_relu",
    "interval_sigmoid",
    "interval_tanh",
]

import torch
import torch.nn.functional as F
from torch import Tensor


def interval_relu(
    x_l: Tensor,
    x_u: Tensor,
) -> tuple[Tensor, Tensor]:
    y_l = x_l.clamp_min(0)
    y_u = x_u.clamp_min(0)

    return y_l, y_u


def interval_sigmoid(
    x_l: Tensor,
    x_u: Tensor,
) -> tuple[Tensor, Tensor]:
    y_l = F.sigmoid(x_l)
    y_u = F.sigmoid(x_u)

    y_l = torch.clamp(y_l, 0.0, 1.0)
    y_u = torch.clamp(y_u, 0.0, 1.0)

    return y_l, y_u


def interval_tanh(
    x_l: Tensor,
    x_u: Tensor,
) -> tuple[Tensor, Tensor]:
    y_l = F.tanh(x_l)
    y_u = F.tanh(x_u)

    y_l = torch.clamp(y_l, -1.0, 1.0)
    y_u = torch.clamp(y_u, -1.0, 1.0)

    return y_l, y_u
