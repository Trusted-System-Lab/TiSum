__all__ = ["add_constrs_maxpool2d"]

import itertools

import mip
import numpy as np
from mip import Model, Var, Constr, ConstraintPriority


def _add_lp_constrs_nontriv_maxpool(
    constrs: list[Constr],
    pool_x: list[Var],
    y_i: Var,
    pool_ub: list[float],
    *,
    name: str,
    priority: ConstraintPriority,
):
    model = y_i.model
    for i, x_i in enumerate(pool_x):
        constr = model.add_constr(
            y_i >= x_i,
            name=f"{name}_l{i}",
            priority=priority,
        )
        constrs.append(constr)

    constr = model.add_constr(
        y_i <= max(pool_ub),
        name=f"{name}_u",
        priority=priority,
    )
    constrs.append(constr)


def _add_mip_constrs_nontriv_maxpool(
    constrs: list[Constr],
    pool_x: list[Var],
    y_i: Var,
    pool_lb: list[float],
    pool_ub: list[float],
    *,
    name: str,
    priority: ConstraintPriority,
):
    model = y_i.model
    n = len(pool_x)

    z = [model.add_var(name=f"{name}_z{i}", var_type=mip.BINARY) for i in range(n)]

    constr = model.add_constr(
        mip.xsum(z) == 1,
        name=f"{name}_zsum",
        priority=priority,
    )
    constrs.append(constr)

    l_max = max(pool_lb)
    u_max = max(pool_ub)
    for i, (x_i, l_i, u_i) in enumerate(zip(pool_x, pool_lb, pool_ub)):
        M_ub = u_max - l_i
        M_lb = u_i - l_max

        constr = model.add_constr(
            y_i <= x_i + (1 - z[i]) * M_ub,
            name=f"{name}_ub{i}",
            priority=priority,
        )
        constrs.append(constr)

        constr = model.add_constr(
            y_i >= x_i - (1 - z[i]) * M_lb,
            name=f"{name}_lb{i}",
            priority=priority,
        )
        constrs.append(constr)


def add_constrs_maxpool2d(
    model: Model,
    x: list[Var],
    y: list[Var],
    x_lb: np.ndarray,
    x_ub: np.ndarray,
    *,
    input_size: tuple[int, int, int],
    output_size: tuple[int, int, int],
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    name: str = "",
    priority: ConstraintPriority = None,
    use_mip: bool = False,
) -> list[Constr]:
    """Add LP relaxation constraints for MaxPool2d operation."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    if type(stride) is int:
        stride = (stride, stride)
    if type(padding) is int:
        padding = (padding, padding)
    if type(dilation) is int:
        dilation = (dilation, dilation)

    in_c, in_h, in_w = input_size
    out_c, out_h, out_w = output_size
    kernel_h, kernel_w = kernel_size
    assert in_c == out_c

    if not name:
        name = f"gemm"

    constrs = []
    for c, oh, ow in itertools.product(range(out_c), range(out_h), range(out_w)):
        y_idx = c * out_h * out_w + oh * out_w + ow
        y_var = y[y_idx]
        cname = f"{name}_{y_var.name}"

        pool_x = []
        pool_lb = []
        pool_ub = []
        for kh, kw in itertools.product(range(kernel_h), range(kernel_w)):
            ih = oh * stride[0] - padding[0] + kh * dilation[0]
            iw = ow * stride[1] - padding[1] + kw * dilation[1]

            if not (0 <= ih < in_h and 0 <= iw < in_w):
                continue
            x_idx = c * in_h * in_w + ih * in_w + iw

            pool_x.append(x[x_idx])
            pool_lb.append(x_lb[x_idx])
            pool_ub.append(x_ub[x_idx])

        if len(pool_x) == 0:
            constr = model.add_constr(
                y_var == 0,
                name=f"{cname}_t",
                priority=priority,
            )
            constrs.append(constr)
            continue

        top1_ub = float("-inf")
        top2_ub = float("-inf")
        top1_lb = float("inf")
        top1_idx = None
        for i in range(len(pool_x)):
            lb_i = pool_lb[i]
            ub_i = pool_ub[i]
            if ub_i > top1_ub:
                top2_ub = top1_ub
                top1_ub = ub_i
                top1_idx = i
                top1_lb = lb_i
            elif ub_i > top2_ub:
                top2_ub = ub_i
        if top2_ub < top1_lb:
            constr = model.add_constr(
                y_var == pool_x[top1_idx],
                name=f"{cname}_t",
                priority=priority,
            )
            constrs.append(constr)
            continue

        if use_mip:
            _add_mip_constrs_nontriv_maxpool(
                constrs,
                pool_x,
                y_var,
                pool_lb,
                pool_ub,
                name=cname,
                priority=priority,
            )
        else:
            _add_lp_constrs_nontriv_maxpool(
                constrs,
                pool_x,
                y_var,
                pool_lb,
                name=cname,
                priority=priority,
            )

    return constrs
