__all__ = ["linear_relax_sigmoid", "linear_relax_tanh"]

from typing import Callable

import torch
from torch import Tensor

from ..._enums import RelaxMode
from ...functions import sigmoid, dsigmoid, tanh, dtanh


def _linear_relax_deepzono(
    x_l: Tensor,
    x_u: Tensor,
    y_l: Tensor,
    y_u: Tensor,
    k_l: Tensor,
    k_u: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute the linear relaxation of a function using the DeepZono method."""
    s_l = torch.minimum(k_l, k_u)
    s_u = s_l.clone()
    t_l = -s_l * x_l + y_l
    t_u = -s_u * x_u + y_u
    return s_l.detach_(), t_l.detach_(), s_u.detach_(), t_u.detach_()


def _linear_relax_deeppoly(
    x_l: Tensor,
    x_u: Tensor,
    y_l: Tensor,
    y_u: Tensor,
    k_l: Tensor,
    k_u: Tensor,
    k_lu: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute the linear relaxation of a function using the DeepPoly method."""
    mask0: Tensor = x_u == x_l
    mask1: Tensor = (x_u <= 0) & ~mask0
    mask2: Tensor = (x_l >= 0) & ~mask0
    mask3: Tensor = ~(mask0 | mask1 | mask2)

    s_l = torch.where(mask0, k_l, torch.zeros_like(k_l))
    s_u = s_l.clone()

    s_l = torch.where(mask1, k_l, s_l)
    s_u = torch.where(mask1, k_lu, s_u)

    s_l = torch.where(mask2, k_lu, s_l)
    s_u = torch.where(mask2, k_u, s_u)

    s_l = torch.where(mask3, torch.minimum(k_l, k_u), s_l)
    s_u = torch.where(mask3, s_l, s_u)

    t_l = -s_l * x_l + y_l
    t_u = -s_u * x_u + y_u

    return s_l.detach_(), t_l.detach_(), s_u.detach_(), t_u.detach_()


def _linear_relax_sshape(
    x_l: Tensor,
    x_u: Tensor,
    y_l: Tensor,
    y_u: Tensor,
    k_l: Tensor,
    k_u: Tensor,
    k_lu: Tensor,
    x_m: Tensor,
    y_m: Tensor,
    k_m: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute the linear relaxation of an S-shape function."""
    mask0: Tensor = x_u == x_l
    mask1: Tensor = (k_u >= k_lu) & ~mask0
    mask2: Tensor = (k_l >= k_lu) & ~mask0
    mask3: Tensor = ~(mask0 | mask1 | mask2)

    s_l = torch.where(mask0, k_l, 0.0)
    s_u = s_l.clone()
    t_l = torch.where(mask0, y_l - s_l * x_l, 0.0)
    t_u = t_l.clone()

    s_l = torch.where(mask1, k_m, s_l)
    s_u = torch.where(mask1, k_lu, s_u)
    t_l = torch.where(mask1, y_m - s_l * x_m, t_l)
    t_u = torch.where(mask1, y_l - s_u * x_l, t_u)

    s_l = torch.where(mask2, k_lu, s_l)
    s_u = torch.where(mask2, k_m, s_u)
    t_l = torch.where(mask2, y_u - s_l * x_u, t_l)
    t_u = torch.where(mask2, y_m - s_u * x_m, t_u)

    s_l = torch.where(mask3, k_l, s_l)
    s_u = torch.where(mask3, k_u, s_u)
    t_l = torch.where(mask3, y_l - s_l * x_l, t_l)
    t_u = torch.where(mask3, y_u - s_u * x_u, t_u)

    return s_l.detach_(), t_l.detach_(), s_u.detach_(), t_u.detach_()


def linear_relax_sshape(
    x_l: Tensor,
    x_u: Tensor,
    f: Callable,
    df: Callable,
    mode: RelaxMode,
) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:

    y_l, y_u, k_l, k_u = f(x_l), f(x_u), df(x_l), df(x_u)

    if mode == RelaxMode.DEEPZONO:
        return _linear_relax_deepzono(x_l, x_u, y_l, y_u, k_l, k_u)

    k_lu = (y_u - y_l) / (x_u - x_l)

    if mode == RelaxMode.DEEPPOLY:
        return _linear_relax_deeppoly(x_l, x_u, y_l, y_u, k_l, k_u, k_lu)

    x_m = (x_l + x_u) * 0.5
    y_m = f(x_m)
    k_m = df(x_m)

    return _linear_relax_sshape(x_l, x_u, y_l, y_u, k_l, k_u, k_lu, x_m, y_m, k_m)


def linear_relax_sigmoid(
    x_l: Tensor,
    x_u: Tensor,
    *,
    mode: RelaxMode,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_relax_sshape(x_l, x_u, sigmoid, dsigmoid, mode)


def linear_relax_tanh(
    x_l: Tensor,
    x_u: Tensor,
    *,
    mode: RelaxMode,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_relax_sshape(x_l, x_u, tanh, dtanh, mode)
