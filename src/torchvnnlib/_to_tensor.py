__all__ = ["convert_to_tensor"]

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

import torch
from torch import Tensor

from .ast import *


def _convert_input_bounds(expr: And, n_inputs: int) -> Tensor:
    intput_bounds = [[None] * 2 for _ in range(n_inputs)]
    for sub_expr in expr:
        if not (
            isinstance(sub_expr, Leq)
            or isinstance(sub_expr, Geq)
            or isinstance(sub_expr, Eq)
        ):
            raise ValueError(f"Invalid input bound expression: {sub_expr}")
        idx = int(sub_expr.left.name[2:])
        value = float(sub_expr.right.value)
        if isinstance(sub_expr, Leq):
            intput_bounds[idx][1] = value
        elif isinstance(sub_expr, Geq):
            intput_bounds[idx][0] = value
        elif isinstance(sub_expr, Eq):
            intput_bounds[idx][0] = value
            intput_bounds[idx][1] = value
        else:
            raise RuntimeError(f"Invalid {sub_expr} to obtain input bounds.")

    for i in range(n_inputs):
        for j in range(2):
            if intput_bounds[i][j] is None:
                raise ValueError(f"Missing input bound for input {i}: {intput_bounds}")

    return torch.tensor(intput_bounds, dtype=torch.float64)


def _convert_linear_poly(
    constr: Tensor, expr: Expr, y_dim: int, x_dim: int, is_add: bool = True
) -> Tensor:
    """The constraints is a linear polynomial with (x_dim + y_dim + 1) dimensions."""
    if isinstance(expr, Var):
        idx = int(expr.name[2:])
        if "Y" in expr.name:
            constr[idx + 1] += 1 if is_add else -1
        elif "X" in expr.name:
            constr[idx + y_dim + 1] += 1 if is_add else -1
        else:
            raise ValueError(f"Invalid variable name: {expr.name}")
    elif isinstance(expr, Cst):
        constr[0] += expr.value if is_add else -expr.value
    elif isinstance(expr, Add):
        for sub_expr in expr.args:
            _convert_linear_poly(constr, sub_expr, y_dim, x_dim, is_add)
    elif isinstance(expr, Mul):
        left = expr.left
        right = expr.right
        if isinstance(left, Cst) and isinstance(right, Var):
            idx = int(right.name[2:])
            if "Y" in right.name:
                constr[idx + 1] += left.value if is_add else -left.value
            elif "X" in right.name:
                constr[idx + y_dim + 1] += left.value if is_add else -left.value
            else:
                raise ValueError(f"Invalid variable name: {right.name}")
    elif isinstance(expr, Sub):
        left = expr.left
        right = expr.right
        _convert_linear_poly(constr, left, y_dim, x_dim)
        _convert_linear_poly(constr, right, y_dim, x_dim, is_add=False)

    return constr


def _convert_linear_constr(left: Expr, right: Expr, y_dim: int, x_dim: int) -> Tensor:
    """By default, we handle left <= right."""
    constr = torch.zeros(y_dim + 1, dtype=torch.float64)

    def _is_asmd(expr: Expr) -> bool:
        return (
            isinstance(expr, Add)
            or isinstance(expr, Sub)
            or isinstance(expr, Mul)
            or isinstance(expr, Div)
        )

    if isinstance(left, Var):
        idx = int(left.name[2:])
        constr[idx + 1] += -1
    elif isinstance(left, Cst):
        constr[0] += -left.value
    else:
        raise NotImplementedError(f"Now only support Var and Cst for left: {left}")

    if isinstance(right, Var):
        idx = int(right.name[2:])
        constr[idx + 1] += 1
    elif isinstance(right, Cst):
        constr[0] += right.value
    elif _is_asmd(right):
        extended_constr = torch.zeros(y_dim + x_dim + 1, dtype=torch.float64)
        extended_constr[: y_dim + 1] = constr
        constr = extended_constr
        constr = _convert_linear_poly(constr, right, y_dim, x_dim)
    else:
        raise NotImplementedError(f"Now only support Var and Cst for right: {right}")

    return constr


def _convert_and_output_constrs(expr: And, n_outputs: int, n_inputs: int) -> Tensor:
    """We treat all constraints in the form of b + Ax >= 0."""
    y_dim = n_outputs
    x_dim = n_inputs
    output_constrs = []
    for i, sub_expr in enumerate(expr):
        left = sub_expr.left
        right = sub_expr.right

        if isinstance(sub_expr, Leq):
            constr = _convert_linear_constr(left, right, y_dim, x_dim)
        elif isinstance(sub_expr, Geq):
            constr = _convert_linear_constr(left, right, y_dim, x_dim)
            constr = -constr
            constr.masked_fill_(constr == 0.0, 0.0)
        else:
            raise ValueError(f"Invalid output constraint expression: {sub_expr}")

        output_constrs.append(constr)
    output_constrs = torch.stack(output_constrs)

    return output_constrs


def _convert_output_constrs(expr: Or, n_outputs: int, n_inputs: int) -> list[Tensor]:
    """The output is a list because we may have multiple Or expressions with different number of constraints."""
    or_output_constrs = []
    for and_expr in expr.args:
        and_expr: And
        and_output_constrs = _convert_and_output_constrs(and_expr, n_outputs, n_inputs)
        or_output_constrs.append(and_output_constrs)

    return or_output_constrs


def _convert_one_property(
    expr: And, n_inputs: int, n_outputs: int
) -> tuple[Tensor, list[Tensor]]:
    """Convert one property to a pair of input and output constraints."""
    assert isinstance(expr, And), f"Expected And expression, got {type(expr)}"
    input_bounds_expr: And
    output_constrs_expr: Or
    input_bounds_expr = expr.args[0]
    output_constrs_expr = expr.args[1]

    input_bounds = _convert_input_bounds(input_bounds_expr, n_inputs)
    output_constrs = _convert_output_constrs(output_constrs_expr, n_outputs, n_inputs)

    return input_bounds, output_constrs


def convert_to_tensor(
    expr: And, n_inputs: int, n_outputs: int
) -> list[list[tuple[Tensor, list[Tensor]]]]:
    """Now we should get an Expr."""


    def _process_or_expr(
        or_expr: Or, n_inputs: int, n_outputs: int
    ) -> list[tuple[Any, Any]]:
        if not isinstance(or_expr, Or):
            raise ValueError(f"Expected Or expression, got {type(or_expr)}")

        convert = partial(_convert_one_property, n_inputs=n_inputs, n_outputs=n_outputs)

        with ThreadPoolExecutor() as executor:
            or_groups = list(executor.map(convert, or_expr.args))

        return or_groups

    def convert_all_properties_parallel(expr, n_inputs: int, n_outputs: int):
        process = partial(_process_or_expr, n_inputs=n_inputs, n_outputs=n_outputs)

        with ThreadPoolExecutor() as executor:
            and_properties = list(executor.map(process, expr.args))

        return and_properties

    return convert_all_properties_parallel(expr, n_inputs, n_outputs)
