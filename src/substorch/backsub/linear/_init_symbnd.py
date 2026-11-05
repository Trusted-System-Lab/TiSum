__all__ = [
    "init_symbnd_similar_eye",
    "init_symbnd_avgpool2d",
    "init_symbnd_batchnorm2d",
    "init_symbnd_conv2d",
    "init_symbnd_convtranspose2d",
    "init_symbnd_gemm",
    "init_symbnd_matmulcst",
    "init_symbnd_reducemean",
    "init_symbnd_reducsum",
    "init_symbnd_resize",
    "init_symbnd_scatter",
    "init_symbnd_scatternd",
    "init_symbnd_upsample",
]

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def init_symbnd_similar_eye(
    output_size: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    symbnd_coeffs = torch.eye(math.prod(output_size), dtype=dtype, device=device)
    symbnd_coeffs = symbnd_coeffs.reshape(-1, *output_size)
    symbnd_const = torch.zeros(output_size, dtype=dtype, device=device)
    symbnd_const = symbnd_const.flatten()

    return symbnd_coeffs.detach(), symbnd_const.detach()


def init_symbnd_avgpool2d(
    output_size: tuple[int, int, int],
    *,
    kernel_size: int | tuple[int, int],
    padding: int | tuple[int, int] = 0,
    stride: int | tuple[int, int] = 1,
    output_padding: int | tuple[int, int] = 0,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    channel = output_size[0]
    n = math.prod(output_size)
    weight_shape = (channel, channel, *kernel_size)
    weight_value = 1 / math.prod(kernel_size)
    weight = torch.full(weight_shape, weight_value, dtype=dtype, device=device)
    symbnd_coeffs = torch.eye(n, dtype=dtype, device=device).reshape(n, *output_size)
    symbnd_const = torch.zeros(n, dtype=dtype, device=device)
    symbnd_coeffs = F.conv_transpose2d(
        symbnd_coeffs,
        weight,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )

    return symbnd_coeffs.detach_(), symbnd_const.detach_()


def init_symbnd_batchnorm2d(
    output_size: tuple[int, int, int],
    *,
    weight: Tensor,
    bias: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    real_weight = weight / torch.sqrt(running_var + eps)
    real_bias = bias - running_mean * real_weight

    symbnd_coeffs, symbnd_const = init_symbnd_similar_eye(
        output_size, dtype=weight.dtype, device=weight.device
    )

    b, c, h, w = symbnd_coeffs.shape
    symbnd_coeffs[b : b + c] *= real_weight.view(1, -1, 1, 1)
    symbnd_const[b : b + c] += real_bias.view(1, -1, 1, 1)

    return symbnd_coeffs.detach_(), symbnd_const.detach_()


def init_symbnd_conv2d(
    weight: Tensor,
    bias: Tensor,
    *,
    output_size: tuple[int, int, int],
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    stride: int | tuple[int, int] = 1,
    output_padding: int | tuple[int, int] = 0,
    groups: int = 1,
) -> tuple[Tensor, Tensor]:
    n = math.prod(output_size)
    dtype, device = weight.dtype, weight.device
    symbnd_coeffs = torch.eye(n, dtype=dtype, device=device).reshape(n, *output_size)
    symbnd_const = (symbnd_coeffs * bias.reshape(1, -1, 1, 1)).sum(dim=(1, 2, 3))
    symbnd_coeffs = F.conv_transpose2d(
        symbnd_coeffs,
        weight,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )

    return symbnd_coeffs.detach_(), symbnd_const.detach_()


def init_symbnd_convtranspose2d(
    weight: Tensor,
    bias: Tensor,
    *,
    output_size: tuple[int, int, int],
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    stride: int | tuple[int, int] = 1,
    groups: int = 1,
) -> tuple[Tensor, Tensor]:
    n = math.prod(output_size)
    dtype, device = weight.dtype, weight.device
    symbnd_coeffs = torch.eye(n, dtype=dtype, device=device).reshape(n, *output_size)
    symbnd_const = (symbnd_coeffs * bias.reshape(1, -1, 1, 1)).sum(dim=(1, 2, 3))
    symbnd_coeffs = F.conv2d(
        symbnd_coeffs,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    return symbnd_coeffs.detach_(), symbnd_const.detach_()


def init_symbnd_gemm(
    weight: Tensor,
    bias: Tensor,
) -> tuple[Tensor, Tensor]:
    symbnd_coeffs = weight.t().clone()
    symbnd_const = bias.clone()

    return symbnd_coeffs.detach_(), symbnd_const.detach_()


def init_symbnd_matmulcst(
    weight: Tensor,
) -> tuple[Tensor, Tensor]:
    symbnd_coeffs = weight.t().clone()
    symbnd_const = torch.zeros(
        weight.shape[1], dtype=weight.dtype, device=weight.device
    )
    return symbnd_coeffs.detach_(), symbnd_const.detach_()


def init_symbnd_reducemean(
    ori_shape: tuple[int, ...],
    *,
    dim: int | tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if ori_shape == ():
        ori_shape = (1,)

    output_shape = list(ori_shape)
    repeat_shape = [1] * len(ori_shape)
    if isinstance(dim, int):
        dim = (dim,)
    for d in dim:
        output_shape[d] = 1
        repeat_shape[d] = ori_shape[d]

    n = math.prod(output_shape)
    symbnd_coeffs = torch.eye(n, dtype=dtype, device=device).reshape(-1, *output_shape)
    symbnd_const = torch.zeros(n, dtype=dtype, device=device)
    symbnd_coeffs = symbnd_coeffs.repeat(1, *repeat_shape)
    dim = [d + 1 for d in dim]
    symbnd_coeffs[:, dim] /= math.prod(repeat_shape)

    return symbnd_coeffs.detach_(), symbnd_const.detach_()


def init_symbnd_reducsum(
    ori_shape: tuple[int, ...],
    *,
    dim: int | tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if ori_shape == ():
        ori_shape = (1,)
    output_shape = list(ori_shape)
    repeat_shape = [1] * len(ori_shape)
    if isinstance(dim, int):
        dim = (dim,)
    for d in dim:
        output_shape[d] = 1
        repeat_shape[d] = ori_shape[d]

    n = math.prod(output_shape)
    symbnd_coeffs = torch.eye(n, dtype=dtype, device=device).reshape(-1, *output_shape)
    symbnd_const = torch.zeros(n, dtype=dtype, device=device)
    symbnd_coeffs = symbnd_coeffs.repeat(1, *repeat_shape)

    return symbnd_coeffs.detach_(), symbnd_const.detach_()


def init_symbnd_resize(
    output_size: tuple[int, ...],
    *,
    input_size: tuple[int, ...] = None,
    scale_factor: tuple[float, ...] = None,
    mode: str = "nearest",
    align_corners: bool = False,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    assert input_size is None, "size should be None."
    assert mode == "nearest", "Only nearest mode is supported for now."
    assert align_corners is False, "Only align_corners is supported for now."
    assert scale_factor is not None, "scale_factor should not be None."

    scale_factor = [1.0 / s for s in scale_factor]
    n = math.prod(output_size)
    symbnd_coeffs = torch.eye(n, dtype=dtype, device=device).reshape(-1, *output_size)
    symbnd_coeffs = F.interpolate(
        symbnd_coeffs,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
    )

    symbnd_const = torch.zeros(n, dtype=dtype, device=device)

    return symbnd_coeffs.detach_(), symbnd_const.detach_()


def init_symbnd_scatter(
    input_size: tuple[int, ...],
    *,
    dim: int,
    index: Tensor,
    src: Tensor,
) -> tuple[Tensor, Tensor]:
    raise NotImplementedError


def init_symbnd_scatternd(
    input_size: tuple[int, ...],
    *,
    index: Tensor,
    src: Tensor,
) -> tuple[Tensor, Tensor]:
    raise NotImplementedError


def init_symbnd_upsample(
    output_size: tuple[int, ...],
    *,
    scale_factor: tuple[float, ...] = None,
    mode: str = "nearest",
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    assert mode == "nearest", "Only nearest mode is supported for now."
    assert scale_factor is not None, "scale_factor should not be None."

    return init_symbnd_resize(
        output_size,
        scale_factor=scale_factor,
        mode=mode,
        dtype=dtype,
        device=device,
    )
