__all__ = ["ada_relax_sigmoid", "ada_relax_tanh"]

from typing import Callable

import torch
from torch import Tensor

from ._update_rlx import normalize_param
from ...functions import sigmoid, dsigmoid, tanh, dtanh
from ...tangent_lines import (
    cal_second_tangent_line_sigmoid,
    cal_second_tangent_line_tanh,
)


def ada_relax_sshape(
    x_l: Tensor,
    x_u: Tensor,
    f: Callable,
    df: Callable,
    cal_second_tangent_line: Callable,
    *,
    ada_mask: Tensor,
) -> tuple[
    tuple[Tensor, Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor],
    tuple[tuple[Tensor, Tensor] | None, tuple[Tensor, Tensor] | None],
    tuple[Tensor | None, Tensor | None],
]:
    dtype, device = x_l.dtype, x_l.device
    y_l, y_u, k_l, k_u = f(x_l), f(x_u), df(x_l), df(x_u)
    k_lu = (y_u - y_l) / (x_u - x_l)
    x_m = (x_l + x_u) * 0.5
    y_m, k_m = f(x_m), df(x_m)

    mask0: Tensor = x_u == x_l
    mask1: Tensor = (k_u >= k_lu) & ~mask0
    mask2: Tensor = (k_l >= k_lu) & ~mask0
    mask3: Tensor = ~(mask0 | mask1 | mask2)

    s_l = torch.where(mask0, k_l, 0.0)
    s_u = s_l.clone()
    t_l = torch.where(mask0, y_l - s_l * x_l, 0.0)
    t_u = t_l.clone()

    s_l = torch.where(mask1, k_m, s_l)
    t_l = torch.where(mask1, y_m - s_l * x_m, t_l)
    s_u = torch.where(mask1, k_lu, s_u)
    t_u = torch.where(mask1, y_l - s_u * x_l, t_u)

    s_l = torch.where(mask2, k_lu, s_l)
    t_l = torch.where(mask2, y_u - s_l * x_u, t_l)
    s_u = torch.where(mask2, k_m, s_u)
    t_u = torch.where(mask2, y_m - s_u * x_m, t_u)

    s_l = torch.where(mask3, k_l, s_l)
    t_l = torch.where(mask3, y_l - s_l * x_l, t_l)
    s_u = torch.where(mask3, k_u, s_u)

    t_u = torch.where(mask3, y_u - s_u * x_u, t_u)

    s_l = s_l.detach_()
    t_l = t_l.detach_()
    s_u = s_u.detach_()
    t_u = t_u.detach_()

    idx = torch.arange(ada_mask.numel(), dtype=torch.long, device=device)
    lower_ada_idx = torch.nonzero(mask1 | mask3).squeeze()
    upper_ada_idx = torch.nonzero(mask2 | mask3).squeeze()

    lower_ada_range = None
    upper_ada_range = None
    lower_param = None
    upper_param = None
    if lower_ada_idx.numel() > 0:
        _, _, lower_tangent_point = cal_second_tangent_line(
            x1=x_u[lower_ada_idx], get_big=False
        )
        lower_ada_range = (x_l[lower_ada_idx], lower_tangent_point)
        lower_param = torch.zeros_like(lower_ada_idx, dtype=dtype, device=device)
        lower_param[idx[mask1]] = x_m[mask1]
        lower_param[idx[mask3]] = x_l[mask3]
        lower_param = normalize_param(lower_param, lower_ada_range)
        lower_ada_idx = lower_ada_idx.clone().detach()
        lower_ada_range = (
            lower_ada_range[0].clone().detach(),
            lower_ada_range[1].clone().detach(),
        )
        lower_param = lower_param.clone().detach().requires_grad_(True)

    if upper_ada_idx.numel() > 0:
        _, _, upper_tangent_point = cal_second_tangent_line(
            x1=x_l[upper_ada_idx], get_big=True
        )
        upper_ada_range = (upper_tangent_point, x_u[upper_ada_idx])
        upper_param = torch.zeros_like(upper_ada_idx, dtype=dtype, device=device)
        upper_param[idx[mask2]] = x_m[mask2]
        upper_param[idx[mask3]] = x_u[mask3]
        upper_param = normalize_param(upper_param, upper_ada_range)
        upper_ada_idx = upper_ada_idx.clone().detach()
        upper_ada_range = (
            upper_ada_range[0].clone().detach(),
            upper_ada_range[1].clone().detach(),
        )
        upper_param = upper_param.clone().detach().requires_grad_(True)

    return (
        (s_l, t_l, s_u, t_u),
        (lower_ada_idx, upper_ada_idx),
        (lower_ada_range, upper_ada_range),
        (lower_param, upper_param),
    )


def ada_relax_sigmoid(
    x_l: Tensor,
    x_u: Tensor,
    *,
    ada_mask: Tensor,
) -> tuple[
    tuple[Tensor, Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor],
    tuple[tuple[Tensor, Tensor] | None, tuple[Tensor, Tensor] | None],
    tuple[Tensor | None, Tensor | None],
]:
    return ada_relax_sshape(
        x_l,
        x_u,
        sigmoid,
        dsigmoid,
        cal_second_tangent_line_sigmoid,
        ada_mask=ada_mask,
    )


def ada_relax_tanh(
    x_l: Tensor,
    x_u: Tensor,
    *,
    ada_mask: Tensor,
) -> tuple[
    tuple[Tensor, Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor],
    tuple[tuple[Tensor, Tensor] | None, tuple[Tensor, Tensor] | None],
    tuple[Tensor | None, Tensor | None],
]:
    return ada_relax_sshape(
        x_l,
        x_u,
        tanh,
        dtanh,
        cal_second_tangent_line_tanh,
        ada_mask=ada_mask,
    )
