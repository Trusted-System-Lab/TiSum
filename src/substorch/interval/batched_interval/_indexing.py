__all__ = ["batched_interval_slice"]

from torch import Tensor

from ...utils import gen_slice_list


def batched_interval_slice(
    x_l: Tensor,
    x_u: Tensor,
    *,
    starts: list[int],
    ends: list[int],
    axes: list[int],
    steps: list[int],
) -> tuple[Tensor, Tensor]:
    dims = x_l.dim() - 1
    slicing = gen_slice_list(dims, starts, ends, axes, steps)
    y_l = x_l[:, *slicing]
    y_u = x_u[:, *slicing]

    return y_l, y_u
