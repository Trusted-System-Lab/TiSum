__all__ = ["sigmoid", "dsigmoid", "ddsigmoid", "tanh", "dtanh"]

import torch.nn.functional as F
from torch import Tensor


def sigmoid(x: Tensor) -> Tensor:
    return F.sigmoid(x)


def dsigmoid(x: Tensor) -> Tensor:
    s = F.sigmoid(x)
    return s * (1.0 - s)


def ddsigmoid(x: Tensor) -> Tensor:
    s = F.sigmoid(x)
    return s * (1.0 - s) * (1.0 - 2.0 * s)


def tanh(x: Tensor) -> Tensor:
    return F.tanh(x)


def dtanh(x: Tensor) -> Tensor:
    return 1.0 - F.tanh(x) ** 2
