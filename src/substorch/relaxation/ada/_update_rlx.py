__all__ = ["update_rlx_by_tangent_line"]

from typing import Callable

from torch import Tensor


def normalize_param(
    param: Tensor,
    input_range: tuple[Tensor, Tensor],
) -> Tensor:
    return (param - input_range[0]) / (input_range[1] - input_range[0])


def update_rlx_by_tangent_line(
    param: Tensor,
    input_range: tuple[Tensor, Tensor],
    f: Callable,
    df: Callable,
) -> tuple[Tensor, Tensor]:
    param = param.clamp(0, 1).clone().detach()
    param.requires_grad_(True)

    param = param * (input_range[1] - input_range[0]) + input_range[0]

    s = df(param)
    t = f(param) - s * param

    return s, t
