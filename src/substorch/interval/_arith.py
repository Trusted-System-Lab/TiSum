__all__ = [
    "interval_add",
    "interval_addcst",
]

import torch
from torch import Tensor


def interval_add(
    x_l1: Tensor,
    x_u1: Tensor,
    x_l2: Tensor,
    x_u2: Tensor,
) -> tuple[Tensor, Tensor]:
    y_l = x_l1 + x_l2
    y_u = x_u1 + x_u2

    return y_l, y_u


def interval_addcst(
    x_l: Tensor,
    x_u: Tensor,
    *,
    constant: float | Tensor,
) -> tuple[Tensor, Tensor]:
    y_l = x_l + constant
    y_u = x_u + constant

    return y_l, y_u
