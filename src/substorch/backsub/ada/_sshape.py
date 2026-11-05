__all__ = [
    "ada_backsub_sigmoid",
    "ada_backsub_sigmoid_lower",
    "ada_backsub_tanh",
    "ada_backsub_tanh_lower",
]

from typing import Callable

from torch import Tensor

from ..linear import linear_backsub_unary, linear_backsub_unary_lower
from ...functions import sigmoid, dsigmoid, tanh, dtanh
from ...relaxation.ada._update_rlx import update_rlx_by_tangent_line


def ada_backsub_sshape(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    f: Callable,
    df: Callable,
    ada_idx: tuple[Tensor, Tensor],
    ada_range: tuple[tuple[Tensor, Tensor] | None, tuple[Tensor, Tensor] | None],
    ada_param: tuple[Tensor | None, Tensor | None],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    lower_ada_idx, upper_ada_idx = ada_idx
    lower_ada_range, upper_ada_range = ada_range
    lower_param, upper_param = ada_param
    if lower_param is not None:
        lower_rlx_coeffs[lower_ada_idx], lower_rlx_const[lower_ada_idx] = (
            update_rlx_by_tangent_line(
                lower_param, input_range=lower_ada_range, f=f, df=df
            )
        )
    if upper_param is not None:
        upper_rlx_coeffs[upper_ada_idx], upper_rlx_const[upper_ada_idx] = (
            update_rlx_by_tangent_line(
                upper_param, input_range=upper_ada_range, f=f, df=df
            )
        )

    return linear_backsub_unary(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
    )


def ada_backsub_sshape_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    f: Callable,
    df: Callable,
    ada_idx: tuple[Tensor, Tensor],
    ada_range: tuple[tuple[Tensor, Tensor] | None, tuple[Tensor, Tensor] | None],
    ada_param: tuple[Tensor | None, Tensor | None],
) -> tuple[Tensor, Tensor]:
    lower_ada_idx, _ = ada_idx
    lower_ada_range, _ = ada_range
    lower_param, _ = ada_param
    if lower_param is not None:
        lower_rlx_coeffs[lower_ada_idx], lower_rlx_const[lower_ada_idx] = (
            update_rlx_by_tangent_line(
                lower_param, input_range=lower_ada_range, f=f, df=df
            )
        )

    return linear_backsub_unary_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
    )


def ada_backsub_sigmoid(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    ada_idx: tuple[Tensor, Tensor],
    ada_range: tuple[tuple[Tensor, Tensor] | None, tuple[Tensor, Tensor] | None],
    ada_param: tuple[Tensor | None, Tensor | None],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return ada_backsub_sshape(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        f=sigmoid,
        df=dsigmoid,
        ada_idx=ada_idx,
        ada_range=ada_range,
        ada_param=ada_param,
    )


def ada_backsub_sigmoid_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    ada_idx: tuple[Tensor, Tensor],
    ada_range: tuple[tuple[Tensor, Tensor] | None, tuple[Tensor, Tensor] | None],
    ada_param: tuple[Tensor | None, Tensor | None],
) -> tuple[Tensor, Tensor]:
    return ada_backsub_sshape_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        f=sigmoid,
        df=dsigmoid,
        ada_idx=ada_idx,
        ada_range=ada_range,
        ada_param=ada_param,
    )


def ada_backsub_tanh(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    ada_idx: tuple[Tensor, Tensor],
    ada_range: tuple[tuple[Tensor, Tensor] | None, tuple[Tensor, Tensor] | None],
    ada_param: tuple[Tensor | None, Tensor | None],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return ada_backsub_sshape(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        f=tanh,
        df=dtanh,
        ada_idx=ada_idx,
        ada_range=ada_range,
        ada_param=ada_param,
    )


def ada_backsub_tanh_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    ada_idx: tuple[Tensor, Tensor],
    ada_range: tuple[tuple[Tensor, Tensor] | None, tuple[Tensor, Tensor | None]],
    ada_param: tuple[Tensor | None, Tensor | None],
) -> tuple[Tensor, Tensor]:
    return ada_backsub_sshape_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        f=tanh,
        df=dtanh,
        ada_idx=ada_idx,
        ada_range=ada_range,
        ada_param=ada_param,
    )
