__all__ = [
    "batched_interval_reshape",
    "batched_interval_flatten",
]

from torch import Tensor


def batched_interval_reshape(
    x_l: Tensor,
    x_u: Tensor,
    *,
    ori_shape: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    t = x_l.shape[0]
    y_l = x_l.reshape(t, *ori_shape)
    y_u = x_u.reshape(t, *ori_shape)

    return y_l, y_u


def batched_interval_flatten(
    x_l: Tensor,
    x_u: Tensor,
    *,
    ori_shape: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    return batched_interval_reshape(x_l, x_u, ori_shape=ori_shape)
