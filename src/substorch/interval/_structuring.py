__all__ = ["interval_concat"]

import torch
from torch import Tensor


def interval_concat(
    x_l_list: list[Tensor], x_u_list: list[Tensor], dim: int
) -> tuple[Tensor, Tensor]:
    y_l = torch.cat(x_l_list, dim=dim)
    y_u = torch.cat(x_u_list, dim=dim)

    return y_l, y_u
