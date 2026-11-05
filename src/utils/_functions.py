"""This module implements the activation functions and their derivatives under different data types, including Python float, Numpy, and PyTorch."""

__all__ = [
    "relu",
    "drelu",
    "sigmoid",
    "dsigmoid",
    "ddsigmoid",
    "tanh",
    "dtanh",
    "ddtanh",
    "elu",
    "delu",
    "leakyrelu",
    "dleakyrelu",
    "silu",
    "dsilu",
    "relu_np",
    "drelu_np",
    "sigmoid_np",
    "dsigmoid_np",
    "ddsigmoid_np",
    "tanh_np",
    "dtanh_np",
    "ddtanh_np",
    "elu_np",
    "delu_np",
    "leakyrelu_np",
    "dleakyrelu_np",
    "silu_np",
    "dsilu_np",
    "relu_torch",
    "drelu_torch",
    "sigmoid_torch",
    "dsigmoid_torch",
    "ddsigmoid_torch",
    "tanh_torch",
    "dtanh_torch",
    "ddtanh_torch",
    "elu_torch",
    "delu_torch",
    "leakyrelu_torch",
    "dleakyrelu_torch",
    "silu_torch",
    "dsilu_torch",
]

import numpy as np
import torch
import torch.nn.functional as F
from numpy import ndarray
from torch import Tensor


def relu_np(x: ndarray | float) -> ndarray | float:
    return np.maximum(x, 0.0)


def relu_torch(x: Tensor) -> Tensor:
    return F.relu(x)


def relu(x: Tensor | ndarray | float) -> Tensor | ndarray | float:
    if isinstance(x, Tensor):
        return relu_torch(x)
    return relu_np(x)


def drelu_np(x: ndarray | float) -> ndarray | float:
    return np.where(x > 0, 1.0, 0.0)


def drelu_torch(x: Tensor) -> Tensor:
    return (x > 0).float()


def drelu(x: Tensor | ndarray | float) -> Tensor | ndarray | float:
    if isinstance(x, Tensor):
        return drelu_torch(x)
    return drelu_np(x)


def sigmoid_np(x: ndarray | float) -> ndarray | float:
    return np.reciprocal(1.0 + np.exp(-x))


def sigmoid_torch(x: Tensor) -> Tensor:
    return F.sigmoid(x)


def sigmoid(x: Tensor | ndarray | float) -> Tensor | ndarray | float:
    if isinstance(x, Tensor):
        return sigmoid_torch(x)
    return sigmoid_np(x)


def dsigmoid_np(x: ndarray | float) -> ndarray | float:
    s = sigmoid_np(x)
    return s * (1.0 - s)


def dsigmoid_torch(x: Tensor) -> Tensor:
    s = F.sigmoid(x)
    return s * (1.0 - s)


def dsigmoid(x: Tensor | ndarray | float) -> Tensor | ndarray | float:
    if isinstance(x, Tensor):
        return dsigmoid_torch(x)
    return dsigmoid_np(x)


def ddsigmoid_np(x: ndarray | float) -> ndarray | float:
    s = sigmoid_np(x)
    return s * (1.0 - s) * (1.0 - 2.0 * s)


def ddsigmoid_torch(x: Tensor) -> Tensor:
    s = F.sigmoid(x)
    return s * (1.0 - s) * (1.0 - 2.0 * s)


def ddsigmoid(x: Tensor | ndarray | float) -> Tensor | ndarray | float:
    if isinstance(x, Tensor):
        return ddsigmoid_torch(x)
    return ddsigmoid_np(x)


def tanh_np(x: ndarray | float) -> ndarray | float:
    return np.tanh(x)


def tanh_torch(x: Tensor) -> Tensor:
    return F.tanh(x)


def tanh(x: Tensor | ndarray | float) -> Tensor | ndarray | float:
    if isinstance(x, Tensor):
        return tanh_torch(x)
    return tanh_np(x)


def dtanh_np(x: ndarray | float) -> ndarray | float:
    return 1.0 - np.tanh(x) ** 2


def dtanh_torch(x: Tensor) -> Tensor:
    return 1.0 - F.tanh(x) ** 2


def dtanh(x: Tensor | ndarray | float) -> Tensor | ndarray | float:
    if isinstance(x, Tensor):
        return dtanh_torch(x)
    return dtanh_np(x)


def ddtanh_np(x: ndarray | float) -> ndarray | float:
    return -2.0 * np.tanh(x) * (1.0 - np.tanh(x) ** 2)


def ddtanh_torch(x: Tensor) -> Tensor:
    return -2.0 * F.tanh(x) * (1.0 - F.tanh(x) ** 2)


def ddtanh(x: Tensor | ndarray | float) -> Tensor | ndarray | float:
    if isinstance(x, Tensor):
        return ddtanh_torch(x)
    return ddtanh_np(x)


def elu_np(x: ndarray | float) -> ndarray | float:
    return np.where(x > 0, x, np.exp(x) - 1.0)


def elu_torch(x: Tensor) -> Tensor:
    return F.elu(x)


def elu(x: Tensor | ndarray | float) -> Tensor | ndarray | float:
    if isinstance(x, Tensor):
        return elu_torch(x)
    return elu_np(x)


def delu_np(x: ndarray | float) -> ndarray | float:
    return np.where(x > 0, 1.0, np.exp(x))


def delu_torch(x: Tensor) -> Tensor:
    return torch.where(x > 0, 1.0, torch.exp(x))


def delu(x: Tensor | ndarray | float) -> Tensor | ndarray | float:
    if isinstance(x, Tensor):
        return delu_torch(x)
    return delu_np(x)


def leakyrelu_np(
    x: ndarray | float, negative_slope: ndarray | float
) -> ndarray | float:

    return np.where(x > 0, x, negative_slope * x)


def leakyrelu_torch(x: Tensor, negative_slope: Tensor) -> Tensor:
    return F.leaky_relu(x, negative_slope=negative_slope.item())


def leakyrelu(
    x: Tensor | ndarray | float, negative_slope: Tensor | ndarray | float = 0.01
) -> Tensor | ndarray | float:
    if isinstance(x, Tensor):
        return leakyrelu_torch(x, negative_slope)
    return leakyrelu_np(x, negative_slope)


def dleakyrelu_np(
    x: ndarray | float, negative_slope: ndarray | float
) -> ndarray | float:
    return np.where(x > 0, 1.0, negative_slope)


def dleakyrelu_torch(x: Tensor, negative_slope: Tensor) -> Tensor:
    return torch.where(x > 0, 1.0, negative_slope)


def dleakyrelu(
    x: Tensor | ndarray | float, negative_slope: Tensor | ndarray | float = 0.01
) -> Tensor | ndarray | float:
    if isinstance(x, Tensor):
        return dleakyrelu_torch(x, negative_slope)
    return dleakyrelu_np(x, negative_slope)


def silu_np(x: ndarray | float) -> ndarray | float:
    return np.reciprocal(1.0 + np.exp(-x)) * x


def silu_torch(x: Tensor) -> Tensor:
    return F.silu(x)


def silu(x: Tensor | ndarray | float) -> Tensor | ndarray | float:
    if isinstance(x, Tensor):
        return silu_torch(x)
    return silu_np(x)


def dsilu_np(x: ndarray | float) -> ndarray | float:
    s = sigmoid_np(x)
    return s + x * s * (1.0 - s)


def dsilu_torch(x: Tensor) -> Tensor:
    s = F.sigmoid(x)
    return s + x * s * (1.0 - s)


def dsilu(x: Tensor | ndarray | float) -> Tensor | ndarray | float:
    if isinstance(x, Tensor):
        return dsilu_torch(x)
    return dsilu_np(x)
