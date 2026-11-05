__all__ = ["batched_interval_concat"]

import torch
from torch import Tensor


def batched_interval_concat(
    x_l_list: list[Tensor], x_u_list: list[Tensor], dim: int
) -> tuple[Tensor, Tensor]:
    y_l = torch.cat(x_l_list, dim=dim + 1)
    y_u = torch.cat(x_u_list, dim=dim + 1)

    return y_l, y_u
