"""This module provides the algorithms to numerically calculate the tangent lines of Sigmoid and Tanh functions."""

__all__ = [
    "cal_parallel_tangent_line_tanh_torch",
    "cal_parallel_tangent_line_sigmoid_torch",
    "cal_second_tangent_line_tanh",
    "cal_second_tangent_line_sigmoid",
]


import logging

import torch
import torch.nn.functional as F
from torch import Tensor

from ._exception import NotConverged

_LOG_MIN = 1e-6
_MAX_ITER = 100
_CONVERGE_TOL = 1e-4
_LOG_MIN_TENSOR = torch.tensor(_LOG_MIN, dtype=torch.float64)
_ZERO_TENSOR = torch.tensor(0.0, dtype=torch.float64)

logging.getLogger("numba").setLevel(logging.CRITICAL)


def cal_parallel_tangent_line_sigmoid_torch(
    k: Tensor, get_big: bool
) -> tuple[Tensor, Tensor, Tensor]:
    sign = 1.0 if get_big else -1.0

    temp = torch.maximum(1.0 - 4.0 * k, _ZERO_TENSOR)
    sigma = 2.0 * torch.reciprocal(1.0 + sign * torch.sqrt(temp))
    temp = torch.maximum(sigma - 1.0, _LOG_MIN_TENSOR)
    x = -torch.log(temp)

    b = F.sigmoid(x) - k * x

    return b, k, x


def cal_parallel_tangent_line_tanh_torch(
    k: Tensor, get_big: bool
) -> tuple[Tensor, Tensor, Tensor]:
    sign = 1.0 if get_big else -1.0
    temp = torch.maximum(1.0 - k, _ZERO_TENSOR)
    sigma = sign * torch.sqrt(temp)
    x = torch.log((1.0 + sigma) / (1.0 - sigma)) * 0.5
    b = F.tanh(x) - k * x

    return b, k, x


def cal_second_tangent_line_sigmoid(
    x1: Tensor, get_big: bool, tol: float = _CONVERGE_TOL
) -> tuple[Tensor, Tensor, Tensor]:
    x2 = torch.zeros_like(x1, dtype=x1.dtype, device=x1.device)
    y1 = F.sigmoid(x1)

    for _ in range(_MAX_ITER):
        y2 = F.sigmoid(x2)
        k = (y2 - y1) / (x2 - x1)
        b, k, x_new = cal_parallel_tangent_line_sigmoid_torch(k, get_big)

        if torch.all(torch.abs(x2 - x_new) < tol):
            return b, k, x_new

        x2 = x_new

    raise NotConverged()


def cal_second_tangent_line_tanh(
    x1: Tensor, get_big: bool, tol: float = _CONVERGE_TOL
) -> tuple[Tensor, Tensor, Tensor]:
    x2 = torch.zeros_like(x1, dtype=x1.dtype, device=x1.device)
    y1 = F.tanh(x1)

    for _ in range(_MAX_ITER):
        y2 = F.tanh(x2)
        k = (y2 - y1) / (x2 - x1)
        b, k, x_new = cal_parallel_tangent_line_tanh_torch(k, get_big)

        if torch.all(torch.abs(x2 - x_new) < tol):
            return b, k, x_new

        x2 = x_new

    raise NotConverged()
