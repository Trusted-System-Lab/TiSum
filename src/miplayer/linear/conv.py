__all__ = ["add_constrs_conv2d"]

import itertools

from mip import Model, Var, Constr, ConstraintPriority
from numpy import ndarray


def add_constrs_conv2d(
    model: Model,
    x: list[Var],
    y: list[Var],
    weight: ndarray,
    bias: ndarray | None = None,
    *,
    input_size: tuple[int, int, int],
    output_size: tuple[int, int, int],
    stride: int | tuple[int, int] = (1, 1),
    padding: int | tuple[int, int] = (0, 0),
    dilation: int | tuple[int, int] = (1, 1),
    groups: int = 1,
    name: str = "",
    priority: ConstraintPriority | None = None,
) -> list[Constr]:
    if type(stride) is int:
        stride = (stride, stride)
    if type(padding) is int:
        padding = (padding, padding)
    if type(dilation) is int:
        dilation = (dilation, dilation)

    in_c, in_h, in_w = input_size
    out_c, out_h, out_w = output_size
    w_cout, w_cin, w_kh, w_kw = weight.shape

    assert out_c == w_cout
    assert in_c == w_cin * groups

    if not name:
        name = f"gemm"

    constrs = []
    for oc in range(out_c):
        group_idx = oc // (out_c // groups)
        ic_start = group_idx * (in_c // groups)
        ic_end = ic_start + (in_c // groups)
        for oh, ow in itertools.product(range(out_h), range(out_w)):
            y_idx = oc * out_h * out_w + oh * out_w + ow

            wxb = []
            for ic, kh, kw in itertools.product(
                range(ic_start, ic_end), range(w_kh), range(w_kw)
            ):
                ih = oh * stride[0] - padding[0] + kh * dilation[0]
                iw = ow * stride[1] - padding[1] + kw * dilation[1]
                if not (0 <= ih < in_h and 0 <= iw < in_w):
                    continue
                x_idx = ic * in_h * in_w + ih * in_w + iw
                w_var = weight[oc, ic - ic_start, kh, kw]
                wxb.append(w_var * x[x_idx])

            y_var = y[y_idx]

            if len(wxb) == 0:
                constr = model.add_constr(
                    y_var == 0,
                    name=f"{name}_{y_var.name}",
                    priority=priority,
                )
                constrs.append(constr)
                continue

            if bias is not None:
                wxb.append(bias[oc])
            constr = model.add_constr(
                mip.xsum(wxb) == y_var,
                name=f"{name}_{y_var.name}",
                priority=priority,
            )
            constrs.append(constr)

    return constrs
