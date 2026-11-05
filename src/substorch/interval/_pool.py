__all__ = [
    "interval_conv2d",
    "interval_maxpool2d",
]

import torch
import torch.nn.functional as F
from torch import Tensor


def interval_conv2d(
    x_l: Tensor,
    x_u: Tensor,
    weight: Tensor,
    bias: Tensor,
    *,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> tuple[Tensor, Tensor]:
    w_p = weight.clamp_min(0)
    w_n = weight.clamp_max(0)

    kwargs = {
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
    }
    y_l = F.conv2d(x_l, w_p, bias, **kwargs) + F.conv2d(x_u, w_n, None, **kwargs)
    y_u = F.conv2d(x_u, w_p, bias, **kwargs) + F.conv2d(x_l, w_n, None, **kwargs)

    return y_l, y_u


def interval_maxpool2d(
    x_l: Tensor,
    x_u: Tensor,
    *,
    kernel_size: tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    ceil_mode: bool = False,
) -> tuple[Tensor, Tensor]:
    y_l = F.max_pool2d(
        x_l,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    y_u = F.max_pool2d(
        x_u,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    return y_l, y_u
