__all__ = [
    "linear_backsub_avgpool2d",
    "linear_backsub_avgpool2d_lower",
    "linear_backsub_conv2d",
    "linear_backsub_conv2d_lower",
    "linear_backsub_convtranspose2d",
    "linear_backsub_convtranspose2d_lower",
    "linear_backsub_maxpool2d",
    "linear_backsub_maxpool2d_lower",
]

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def _linear_backsub_conv2d(
    symbnd_coeffs: Tensor,
    symbnd_const: Tensor,
    weight: Tensor,
    bias: Tensor,
    **kwargs,
) -> tuple[Tensor, Tensor]:
    new_symbnd_const = symbnd_const + torch.einsum("bchw, c -> b", symbnd_coeffs, bias)
    new_symbnd_coeffs = F.conv_transpose2d(symbnd_coeffs, weight, **kwargs)
    return new_symbnd_coeffs, new_symbnd_const


def linear_backsub_conv2d(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    weight: Tensor,
    bias: Tensor,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    output_padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    kwargs = {
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "dilation": dilation,
        "groups": groups,
    }
    new_lower_symbnd_coeffs, new_lower_symbnd_const = _linear_backsub_conv2d(
        lower_symbnd_coeffs, lower_symbnd_const, weight, bias, **kwargs
    )
    new_upper_symbnd_coeffs, new_upper_symbnd_const = _linear_backsub_conv2d(
        upper_symbnd_coeffs, upper_symbnd_const, weight, bias, **kwargs
    )

    return (
        new_lower_symbnd_coeffs,
        new_lower_symbnd_const,
        new_upper_symbnd_coeffs,
        new_upper_symbnd_const,
    )


def linear_backsub_conv2d_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    weight: Tensor,
    bias: Tensor,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    output_padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> tuple[Tensor, Tensor]:
    kwargs = {
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "dilation": dilation,
        "groups": groups,
    }
    new_lower_symbnd_coeffs, new_lower_symbnd_const = _linear_backsub_conv2d(
        lower_symbnd_coeffs, lower_symbnd_const, weight, bias, **kwargs
    )

    return new_lower_symbnd_coeffs, new_lower_symbnd_const




def _linear_backsub_avgpool2d(
    symbnd_coeffs: Tensor,
    symbnd_const: Tensor,
    weight: Tensor,
    **kwargs,
) -> tuple[Tensor, Tensor]:
    new_symbnd_const = symbnd_const
    new_symbnd_coeffs = F.conv_transpose2d(symbnd_coeffs, weight, **kwargs)
    return new_symbnd_coeffs, new_symbnd_const


def linear_backsub_avgpool2d(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    output_size: tuple[int, int, int],
    kernel_size: tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    output_padding: int | tuple[int, int] = 0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    dtype = lower_symbnd_coeffs.dtype
    device = lower_symbnd_coeffs.device
    channel = output_size[0]
    weight_shape = (channel, channel, *kernel_size)
    weight_value = 1.0 / math.prod(kernel_size)
    weight = torch.full(weight_shape, weight_value, dtype=dtype, device=device)

    kwargs = {"padding": padding, "stride": stride, "output_padding": output_padding}
    new_lower_symbnd_coeffs, new_lower_symbnd_const = _linear_backsub_avgpool2d(
        lower_symbnd_coeffs, lower_symbnd_const, weight, **kwargs
    )
    new_upper_symbnd_coeffs, new_upper_symbnd_const = _linear_backsub_avgpool2d(
        upper_symbnd_coeffs, upper_symbnd_const, weight, **kwargs
    )

    return (
        new_lower_symbnd_coeffs,
        new_lower_symbnd_const,
        new_upper_symbnd_coeffs,
        new_upper_symbnd_const,
    )


def linear_backsub_avgpool2d_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    output_size: tuple[int, int, int],
    kernel_size: tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    output_padding: int | tuple[int, int] = 0,
) -> tuple[Tensor, Tensor]:
    dtype = lower_symbnd_coeffs.dtype
    device = lower_symbnd_coeffs.device
    channel = output_size[0]
    weight_shape = (channel, channel, *kernel_size)
    weight_value = 1 / math.prod(kernel_size)
    weight = torch.full(weight_shape, weight_value, dtype=dtype, device=device)

    kwargs = {"padding": padding, "stride": stride, "output_padding": output_padding}
    new_lower_symbnd_coeffs, new_lower_symbnd_const = _linear_backsub_avgpool2d(
        lower_symbnd_coeffs, lower_symbnd_const, weight, **kwargs
    )

    return new_lower_symbnd_coeffs, new_lower_symbnd_const


def _linear_backsub_convtranspose2d(
    symbnd_coeffs: Tensor,
    symbnd_const: Tensor,
    weight: Tensor,
    bias: Tensor,
    **kwargs,
) -> tuple[Tensor, Tensor]:
    new_symbnd_const = symbnd_const + torch.einsum("bchw, c -> b", symbnd_coeffs, bias)
    new_symbnd_coeffs = F.conv2d(symbnd_coeffs, weight, **kwargs)
    return new_symbnd_coeffs, new_symbnd_const


def linear_backsub_convtranspose2d(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    weight: Tensor,
    bias: Tensor,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    kwargs = {
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
    }
    new_lower_symbnd_coeffs, new_lower_symbnd_const = _linear_backsub_convtranspose2d(
        lower_symbnd_coeffs, lower_symbnd_const, weight, bias, **kwargs
    )
    new_upper_symbnd_coeffs, new_upper_symbnd_const = _linear_backsub_convtranspose2d(
        upper_symbnd_coeffs, upper_symbnd_const, weight, bias, **kwargs
    )

    return (
        new_lower_symbnd_coeffs,
        new_lower_symbnd_const,
        new_upper_symbnd_coeffs,
        new_upper_symbnd_const,
    )


def linear_backsub_convtranspose2d_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    weight: Tensor,
    bias: Tensor,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> tuple[Tensor, Tensor]:
    kwargs = {
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
    }
    new_lower_symbnd_coeffs, new_lower_symbnd_const = _linear_backsub_convtranspose2d(
        lower_symbnd_coeffs, lower_symbnd_const, weight, bias, **kwargs
    )

    return new_lower_symbnd_coeffs, new_lower_symbnd_const


def _linear_backsub_maxpool2d(
    symbnd_coeffs: Tensor,
    symbnd_const: Tensor,
    rlx_coeffs1: Tensor,
    rlx_const1: Tensor,
    rlx_coeffs2: Tensor,
    rlx_const2: Tensor,
    **kwargs,
) -> tuple[Tensor, Tensor]:
    pos_symbnd_coeffs = symbnd_coeffs.clamp_min(0)
    neg_symbnd_coeffs = symbnd_coeffs.clamp_max(0)

    op_rule = "abc, bc->a"
    new_symbnd_const = (
        symbnd_const
        + torch.einsum(op_rule, pos_symbnd_coeffs, rlx_const1)
        + torch.einsum(op_rule, neg_symbnd_coeffs, rlx_const2)
    )

    op_rule = "abc, bcd->abcd"
    new_symbnd_coeffs = (
        (
            torch.einsum(op_rule, pos_symbnd_coeffs, rlx_coeffs1)
            + torch.einsum(op_rule, neg_symbnd_coeffs, rlx_coeffs2)
        )
        .permute(0, 1, 3, 2)
        .flatten(start_dim=1, end_dim=2)
    )

    new_symbnd_coeffs = F.fold(new_symbnd_coeffs, **kwargs)

    return new_symbnd_coeffs, new_symbnd_const


def linear_backsub_maxpool2d(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    input_size: tuple[int, int, int],
    output_size: tuple[int, int, int],
    kernel_size: tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    n_channel, input_height, input_width = input_size
    output_height_width = output_size[1] * output_size[2]
    kernel_height_width = kernel_size[0] * kernel_size[-1]

    shape = -1, n_channel, output_height_width
    lower_symbnd_coeffs = lower_symbnd_coeffs.reshape(shape)
    upper_symbnd_coeffs = upper_symbnd_coeffs.reshape(shape)

    shape = (n_channel, output_height_width, kernel_height_width)
    lower_rlx_coeffs = lower_rlx_coeffs.reshape(shape)
    upper_rlx_coeffs = upper_rlx_coeffs.reshape(shape)

    shape = (n_channel, output_height_width)
    lower_rlx_const = lower_rlx_const.reshape(shape)
    upper_rlx_const = upper_rlx_const.reshape(shape)

    kwargs = {
        "output_size": input_size[1:],
        "kernel_size": kernel_size,
        "dilation": dilation,
        "padding": padding,
        "stride": stride,
    }
    new_lower_symbnd_coeffs, new_lower_symbnd_const = _linear_backsub_maxpool2d(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        **kwargs,
    )
    new_upper_symbnd_coeffs, new_upper_symbnd_const = _linear_backsub_maxpool2d(
        upper_symbnd_coeffs,
        upper_symbnd_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        **kwargs,
    )

    return (
        new_lower_symbnd_coeffs.reshape(-1, *input_size),
        new_lower_symbnd_const.flatten(),
        new_upper_symbnd_coeffs.reshape(-1, *input_size),
        new_upper_symbnd_const.flatten(),
    )


def linear_backsub_maxpool2d_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    input_size: tuple[int, int, int],
    output_size: tuple[int, int, int],
    kernel_size: tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> tuple[Tensor, Tensor]:
    n_channel, input_height, input_width = input_size
    output_height_width = output_size[1] * output_size[2]
    kernel_height_width = kernel_size[0] * kernel_size[-1]

    shape = -1, n_channel, output_height_width
    lower_symbnd_coeffs = lower_symbnd_coeffs.reshape(shape)

    shape = (n_channel, output_height_width, kernel_height_width)
    lower_rlx_coeffs = lower_rlx_coeffs.reshape(shape)
    upper_rlx_coeffs = upper_rlx_coeffs.reshape(shape)

    shape = (n_channel, output_height_width)
    lower_rlx_const = lower_rlx_const.reshape(shape)
    upper_rlx_const = upper_rlx_const.reshape(shape)

    new_lower_symbnd_coeffs, new_lower_symbnd_const = _linear_backsub_maxpool2d(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        output_size=input_size[1:],
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )

    new_lower_symbnd_coeffs = new_lower_symbnd_coeffs.reshape(-1, *input_size)
    new_lower_symbnd_const = new_lower_symbnd_const.flatten()

    return new_lower_symbnd_coeffs, new_lower_symbnd_const
