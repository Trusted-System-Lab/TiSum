__all__ = ["optimize"]

import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from ._expr import *


def _else_recursion(expr: Expr, func: Callable) -> Expr:
    if isinstance(expr, UnaryOp):
        expr.arg = func(expr.arg)
    elif isinstance(expr, BinaryOp):
        expr.left = func(expr.left)
        expr.right = func(expr.right)
    elif isinstance(expr, NaryOp):
        for i in range(len(expr.args)):
            expr.args[i] = func(expr.args[i])
    return expr


def _remove_single_and_or(expr: Expr) -> Expr:
    """Remove single and from the expression tree."""

    if isinstance(expr, And) or isinstance(expr, Or):
        if len(expr.args) == 1 and "Y" not in expr.args[0].__repr__():
            return expr.args[0]
        else:
            for i in range(len(expr.args)):
                expr.args[i] = _remove_single_and_or(expr.args[i])

    return _else_recursion(expr, _remove_single_and_or)


def _simplify_leqgeq(expr: Expr) -> Expr:
    """If there is "a <= b" and "b <= a" in the same level of an and operation, remove one of them, then combine them into a single expression as a == b."""
    if not isinstance(expr, And):
        return _else_recursion(expr, _simplify_leqgeq)

    simplified_args = []
    leq_pairs = {}
    geq_pairs = {}

    for arg in expr.args:
        if isinstance(arg, Leq):
            key = (str(arg.left), str(arg.right))
            leq_pairs[key] = arg
        elif isinstance(arg, Geq):
            key = (str(arg.left), str(arg.right))
            geq_pairs[key] = arg
        else:
            simplified_args.append(_simplify_leqgeq(arg))

    used = set()
    for (a, b), le_expr in leq_pairs.items():
        if (a, b) in geq_pairs:
            simplified_args.append(Eq(le_expr.left, le_expr.right))
            used.add((a, b))
            used.add((a, b))

    for k, v in leq_pairs.items():
        if k not in used:
            simplified_args.append(v)
    for k, v in geq_pairs.items():
        if k not in used:
            simplified_args.append(v)

    return And(simplified_args)


def _get_priority(var: Var) -> int | float:
    """Extract the number from the variable name."""
    var_repr = repr(var)
    priority = 0
    match = re.search(r"([XY])", var_repr)
    if not match:
        return float("inf")
    letter = match.group(1)
    if letter == "Y":
        priority += 1000000
    match = re.search(r"(\d+)", var.__repr__())
    if not match:
        raise ValueError(f"Invalid variable repr: {var.__repr__()}")
    number = match.group(1)
    if not number:
        raise ValueError(f"Invalid variable repr: {var.__repr__()}")
    if int(number) > 100000000:
        warnings.warn(
            "The number in the variable name is greater than 10000. This will result "
            "in incorrect sorting when printing.",
        )
    priority += int(number)

    return priority


def _sort_vars_in_expr(expr: Expr) -> Expr:
    """Sort the variables in the expression based on the number in Var's name."""
    if isinstance(expr, NaryOp):
        expr.args = [_sort_vars_in_expr(arg) for arg in expr.args]
        expr.args = sorted(expr.args, key=lambda x: _get_priority(x))
        return expr

    return _else_recursion(expr, _sort_vars_in_expr)






def fuse_and_and(expr: Expr) -> Expr:
    """Fuse two And expressions into one."""
    if isinstance(expr, And):
        for arg in expr.args:
            if isinstance(arg, And):
                expr.args.remove(arg)
                expr.args.extend(arg.args)

    return _else_recursion(expr, fuse_and_and)


def optimize(expr: Expr) -> Expr:
    is_and = isinstance(expr, And)
    is_or = isinstance(expr, Or)

    if not (is_and or is_or):
        raise ValueError("The expression must be either an And or an Or.")

    def simplify(epr: Expr) -> Expr:
        epr = _remove_single_and_or(epr)
        epr = _simplify_leqgeq(epr)
        return epr

    with ThreadPoolExecutor() as executor:
        expr_list = list(executor.map(simplify, expr.args))

    if is_and:
        expr = And(expr_list)
    elif is_or:
        expr = Or(expr_list)
    else:
        raise ValueError("The expression must be either an And or an Or.")

    expr = _sort_vars_in_expr(expr)

    return expr
