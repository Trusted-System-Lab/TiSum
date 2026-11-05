__all__ = ["add_constrs_add"]


import mip
from mip import Model, Var, Constr, ConstraintPriority


def add_constrs_add(
    model: Model,
    x1: list[Var],
    x2: list[Var],
    y: list[Var],
    *,
    name: str = "",
    priority: ConstraintPriority | None = None,
) -> list[Constr]:
    assert len(x1) == len(x2) == len(y), (
        f"x1 and x2 must have the same length but " f"got" f" {len(x1)} and {len(x2)}"
    )
    m, n = len(x1), len(y)

    if not name:
        name = f"gemm"

    constrs = []
    for i in range(n):
        wxb = mip.xsum(x1[j] + x2[j] for j in range(m))
        constr = model.add_constr(
            wxb == y[i], name=f"{name}_{y[i].name}", priority=priority
        )
        constrs.append(constr)

    return constrs
