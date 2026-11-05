__all__ = [
    "interval_reshape",
    "interval_flatten",
]

from torch import Tensor


def interval_reshape(
    x_l: Tensor,
    x_u: Tensor,
    *,
    ori_shape: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    y_l = x_l.reshape(ori_shape)
    y_u = x_u.reshape(ori_shape)

    return y_l, y_u


def interval_flatten(
    x_l: Tensor,
    x_u: Tensor,
    *,
    ori_shape: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    return interval_reshape(x_l, x_u, ori_shape=ori_shape)
