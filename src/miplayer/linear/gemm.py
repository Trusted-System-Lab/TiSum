__all__ = ["add_constrs_gemm"]


import mip
from mip import Model, Var, Constr, ConstraintPriority
from numpy import ndarray


def add_constrs_gemm(
    model: Model,
    x: list[Var],
    y: list[Var],
    weight: ndarray,
    bias: ndarray | None = None,
    *,
    name: str = "",
    priority: ConstraintPriority | None = None,
) -> list[Constr]:
    m, n = len(x), len(y)
    if m != weight.shape[0] or n != weight.shape[1]:
        raise ValueError(
            f"Weight shape {weight.shape} does not match x length {m} and y length {n}."
        )
    if bias is not None and len(bias) != n:
        raise ValueError(f"Bias length {len(bias)} does not match y length {n}.")

    if not name:
        name = f"gemm"

    weight = weight.T
    constrs = []
    for i in range(n):
        wxb = mip.xsum(weight[i][j] * x[j] for j in range(m))
        if bias is not None:
            wxb += bias[i]

        constr = model.add_constr(
            wxb == y[i], name=f"{name}_{y[i].name}", priority=priority
        )
        constrs.append(constr)

    return constrs
