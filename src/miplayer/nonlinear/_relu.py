__all__ = ["add_constrs_relu"]

import mip
import numpy as np
from mip import Model, Var, Constr, ConstraintPriority


def _add_lp_constrs_nontriv_relu(
    constrs: list[Constr],
    x_i: Var,
    y_i: Var,
    l_i: float,
    u_i: float,
    *,
    name: str,
    priority: ConstraintPriority,
):
    model = y_i.model
    slope = u_i / (u_i - l_i)
    intercept = -l_i * slope
    constr = model.add_constr(
        y_i <= slope * x_i + intercept, name=f"{name}_u", priority=priority
    )
    constrs.append(constr)


def _add_mip_constrs_nontriv_relu(
    constrs: list[Constr],
    x_i: Var,
    y_i: Var,
    l_i: float,
    u_i: float,
    *,
    name: str,
    priority: ConstraintPriority,
):
    model = x_i.model

    z_i = model.add_var(name=f"{y_i.name}_bin", var_type=mip.BINARY)

    constr = model.add_constr(
        y_i <= x_i - l_i * (1 - z_i), name=f"{name}_u1", priority=priority
    )
    constrs.append(constr)

    constr = model.add_constr(y_i <= u_i * z_i, name=f"{name}_u2", priority=priority)
    constrs.append(constr)


def add_constrs_relu(
    model: Model,
    x: list[Var],
    y: list[Var],
    x_lb: np.ndarray,
    x_ub: np.ndarray,
    *,
    name: str = "",
    priority: ConstraintPriority = None,
    max_num_mip_relu: int = 0,
    min_range_mip_relu: float = 0.1,
) -> tuple[list[Constr], int]:
    if not len(x) == len(y) == len(x_lb) == len(x_ub):
        raise ValueError(
            f"x, y, x_lb, and x_ub must have the same length, but got "
            f"{len(x)}, {len(y)}, {len(x_lb)}, and {len(x_ub)}."
        )

    if not name:
        name = f"relu"

    r = x_ub - x_lb
    r[x_lb >= 0] = 0
    r[x_ub <= 0] = 0
    sorted_indices = np.argsort(-r)

    num_mip_relu = 0
    constrs = []
    for i in sorted_indices:
        x_i = x[i]
        y_i = y[i]
        l_i = float(x_lb[i])
        u_i = float(x_ub[i])

        cname = f"{name}_{y_i.name}"
        if l_i >= 0:
            constr = model.add_constr(
                y_i - x_i == 0, name=f"{cname}_t", priority=priority
            )
            constrs.append(constr)
        elif u_i <= 0:
            pass
        else:
            constr = model.add_constr(y_i >= x_i, name=f"{name}_l", priority=priority)
            constrs.append(constr)

            if num_mip_relu < max_num_mip_relu and (u_i - l_i) > min_range_mip_relu:
                _add_mip_constrs_nontriv_relu(
                    constrs, x_i, y_i, l_i, u_i, name=cname, priority=priority
                )
                num_mip_relu += 1
            else:
                _add_lp_constrs_nontriv_relu(
                    constrs, x_i, y_i, l_i, u_i, name=cname, priority=priority
                )

    return constrs, num_mip_relu
