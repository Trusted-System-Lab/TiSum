__all__ = [
    "relax_clip",
    "relax_cos",
    "relax_div",
    "relax_elu",
    "relax_exp",
    "relax_gelu",
    "relax_leakyrelu",
    "relax_matmul",
    "relax_max",
    "relax_maxpool2d",
    "relax_min",
    "relax_mul",
    "relax_pow",
    "relax_relu",
    "relax_reciprocal",
    "relax_sigmoid",
    "relax_silu",
    "relax_sin",
    "relax_softmax",
    "relax_tanh",
]

from torch import Tensor

from .linear import *
from .quad import *
from .._enums import RelaxMode


def relax_clip(
    lower_bound: Tensor,
    upper_bound: Tensor,
    *,
    min_val: float = None,
    max_val: float = None,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if mode == RelaxMode.SINGLE_QUAD:
        raise NotImplementedError(
            f"Quadratic relaxation for Clip is not implemented yet."
        )

    return linear_relax_clip(lower_bound, upper_bound, min_val=min_val, max_val=max_val)


def relax_cos(
    lower_bound: Tensor,
    upper_bound: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if mode == RelaxMode.SINGLE_QUAD:
        raise NotImplementedError(
            f"Quadratic relaxation for Cos is not implemented yet."
        )

    return linear_relax_cos(lower_bound, upper_bound)


def relax_div(
    a_lower_bound: Tensor,
    a_upper_bound: Tensor,
    b_lower_bound: Tensor,
    b_upper_bound: Tensor,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if mode == RelaxMode.SINGLE_QUAD:
        raise NotImplementedError(
            f"Quadratic relaxation for Div is not implemented yet."
        )

    return linear_relax_div(a_lower_bound, a_upper_bound, b_lower_bound, b_upper_bound)


def relax_elu(
    lower_bound: Tensor,
    upper_bound: Tensor,
    *,
    alpha: float = 1.0,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if mode == RelaxMode.SINGLE_QUAD:
        raise NotImplementedError(
            f"Quadratic relaxation for Elu is not implemented yet."
        )

    if alpha != 1.0:
        raise NotImplementedError(
            f"Relaxation for Elu with alpha {alpha} is not implemented yet."
        )

    return linear_relax_elu(lower_bound, upper_bound)


def relax_exp(
    lower_bound: Tensor,
    upper_bound: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if mode == RelaxMode.SINGLE_QUAD:
        raise NotImplementedError(
            f"Quadratic relaxation for Exp is not implemented yet."
        )

    return linear_relax_exp(lower_bound, upper_bound)


def relax_gelu(
    lower_bound: Tensor,
    upper_bound: Tensor,
    *,
    approximate: str = "none",
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if mode == RelaxMode.SINGLE_QUAD:
        raise NotImplementedError(
            f"Quadratic relaxation for Gelu is not implemented yet."
        )

    if approximate != "none":
        raise NotImplementedError(
            f"Relaxation for Gelu with approximate {approximate} is not implemented "
            f"yet."
        )

    return linear_relax_gelu(lower_bound, upper_bound)


def relax_leakyrelu(
    lower_bound: Tensor,
    upper_bound: Tensor,
    *,
    alpha: float = 0.01,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if mode == RelaxMode.SINGLE_QUAD:
        raise NotImplementedError(
            f"Quadratic relaxation for LeakyRelu is not implemented yet."
        )

    if alpha != 0.01:
        raise NotImplementedError(
            f"Relaxation for LeakyRelu with alpha {alpha} is not implemented yet."
        )

    return linear_relax_leakyrelu(lower_bound, upper_bound, alpha=alpha)


def relax_matmul(
    a_lower_bound: Tensor,
    a_upper_bound: Tensor,
    b_lower_bound: Tensor,
    b_upper_bound: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if mode == RelaxMode.SINGLE_QUAD:
        raise NotImplementedError(
            f"Quadratic relaxation for Matmul is not implemented yet."
        )

    return linear_relax_matmul(
        a_lower_bound, a_upper_bound, b_lower_bound, b_upper_bound
    )


def relax_max(
    lower_bound: Tensor,
    upper_bound: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
):
    if mode == RelaxMode.SINGLE_QUAD:
        raise NotImplementedError(
            f"Quadratic relaxation for Max is not implemented yet."
        )

    return linear_relax_max(lower_bound, upper_bound)


def relax_maxpool2d(
    lower_bound: Tensor,
    upper_bound: Tensor,
    *,
    kernel_size: tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    lower_bound_max: Tensor = None,
    lower_bound_argmax: Tensor = None,
    mask_trivial_pool: Tensor = None,
    lower_relu_quad_coeffs: Tensor = None,
    lower_relu_linear_coeffs: Tensor = None,
    lower_relu_const: Tensor = None,
    upper_relu_quad_coeffs: Tensor = None,
    upper_relu_linear_coeffs: Tensor = None,
    upper_relu_const: Tensor = None,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
):
    if lower_bound.dim() != 3 or upper_bound.dim() != 3:
        raise ValueError(
            f"Relaxation if MaxPool only supports 3D tensor (c, h, w), "
            f"but got lower_bound.shape={lower_bound.shape} and "
            f"upper_bound.shape={upper_bound.shape}."
        )

    if mode == RelaxMode.SINGLE_LINEAR:
        return linear_relax_maxpool2d(
            lower_bound,
            upper_bound,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
    if mode == RelaxMode.SINGLE_QUAD:
        assert lower_bound_max is not None
        assert lower_bound_argmax is not None
        assert mask_trivial_pool is not None
        assert lower_relu_quad_coeffs is not None
        assert lower_relu_linear_coeffs is not None
        assert lower_relu_const is not None
        assert upper_relu_quad_coeffs is not None
        assert upper_relu_linear_coeffs is not None
        assert upper_relu_const is not None
        return quad_relax_maxpool2d_after_relu(
            lower_bound,
            upper_bound,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            x_l_max=lower_bound_max,
            x_l_argmax=lower_bound_argmax,
            mask_trivial_pool=mask_trivial_pool,
            lower_relu_quad_coeffs=lower_relu_quad_coeffs,
            lower_relu_linear_coeffs=lower_relu_linear_coeffs,
            lower_relu_const=lower_relu_const,
            upper_relu_quad_coeffs=upper_relu_quad_coeffs,
            upper_relu_linear_coeffs=upper_relu_linear_coeffs,
            upper_relu_const=upper_relu_const,
        )
    raise NotImplementedError(f"{mode} is not implemented yet.")


def relax_min(
    lower_bound: Tensor,
    upper_bound: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
):
    if mode == RelaxMode.SINGLE_QUAD:
        raise NotImplementedError(
            f"Quadratic relaxation for Min is not implemented yet."
        )

    return linear_relax_min(lower_bound, upper_bound)


def relax_mul(
    a_lower_bound: Tensor,
    a_upper_bound: Tensor,
    b_lower_bound: Tensor,
    b_upper_bound: Tensor,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if mode == RelaxMode.SINGLE_QUAD:
        raise NotImplementedError(
            f"Quadratic relaxation for Mul is not implemented yet."
        )

    return linear_relax_mul(a_lower_bound, a_upper_bound, b_lower_bound, b_upper_bound)


def relax_pow(
    lower_bound: Tensor,
    upper_bound: Tensor,
    *,
    exponent: float,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if mode == RelaxMode.SINGLE_QUAD:
        raise NotImplementedError(
            f"Quadratic relaxation for Pow is not implemented yet."
        )

    return linear_relax_pow(lower_bound, upper_bound, exponent=exponent)


def relax_reciprocal(
    lower_bound: Tensor,
    upper_bound: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if mode == RelaxMode.SINGLE_QUAD:
        raise NotImplementedError(
            f"Quadratic relaxation for Reciprocal is not implemented yet."
        )

    return linear_relax_reciprocal(lower_bound, upper_bound)


def relax_relu(
    lower_bound: Tensor,
    upper_bound: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> (
    tuple[Tensor, Tensor, Tensor, Tensor]
    | tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
):
    if mode == RelaxMode.SINGLE_LINEAR:
        return linear_relax_relu(lower_bound, upper_bound)
    if mode == RelaxMode.DEEPZONO:
        return linear_relax_relu_deepzono(lower_bound, upper_bound)
    if mode == RelaxMode.SINGLE_QUAD:
        return quad_relax_relu(lower_bound, upper_bound)
    raise NotImplementedError(f"{mode} is not implemented yet.")


def relax_sigmoid(
    lower_bound: Tensor,
    upper_bound: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> (
    tuple[Tensor, Tensor, Tensor, Tensor]
    | tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
):
    if mode == RelaxMode.SINGLE_LINEAR:
        return linear_relax_sigmoid(lower_bound, upper_bound, mode=mode)
    if mode == RelaxMode.DEEPZONO:
        return linear_relax_sigmoid(lower_bound, upper_bound, mode=mode)
    if mode == RelaxMode.DEEPPOLY:
        return linear_relax_sigmoid(lower_bound, upper_bound, mode=mode)
    if mode == RelaxMode.SINGLE_QUAD:
        return quad_relax_sigmoid(lower_bound, upper_bound)
    raise NotImplementedError(f"{mode} is not implemented yet.")


def relax_silu(
    lower_bound: Tensor,
    upper_bound: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if mode == RelaxMode.SINGLE_QUAD:
        raise NotImplementedError(
            f"Quadratic relaxation for Silu is not implemented yet."
        )
    return linear_relax_silu(lower_bound, upper_bound)


def relax_sin(
    lower_bound: Tensor,
    upper_bound: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if mode == RelaxMode.SINGLE_QUAD:
        raise NotImplementedError(
            f"Quadratic relaxation for Sin is not implemented yet."
        )

    return linear_relax_sin(lower_bound, upper_bound)


def relax_softmax(
    lower_bound: Tensor,
    upper_bound: Tensor,
    *,
    dim: int = None,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if mode == RelaxMode.SINGLE_QUAD:
        raise NotImplementedError(
            f"Quadratic relaxation for Softmax is not implemented yet."
        )

    return linear_relax_softmax(lower_bound, upper_bound, dim=dim)


def relax_tanh(
    lower_bound: Tensor,
    upper_bound: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> (
    tuple[Tensor, Tensor, Tensor, Tensor]
    | tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
):
    if mode == RelaxMode.SINGLE_LINEAR:
        return linear_relax_tanh(lower_bound, upper_bound, mode=mode)
    if mode == RelaxMode.DEEPZONO:
        return linear_relax_tanh(lower_bound, upper_bound, mode=mode)
    if mode == RelaxMode.DEEPPOLY:
        return linear_relax_tanh(lower_bound, upper_bound, mode=mode)
    if mode == RelaxMode.SINGLE_QUAD:
        return quad_relax_tanh(lower_bound, upper_bound)
    raise NotImplementedError(f"{mode} is not implemented yet.")
