__all__ = [
    "backsub_addcst",
    "backsub_addcst_lower",
    "backsub_add",
    "backsub_add_lower",
    "backsub_argmax",
    "backsub_argmax_lower",
    "backsub_avgpool2d",
    "backsub_avgpool2d_lower",
    "backsub_batchnorm2d",
    "backsub_batchnorm2d_lower",
    "backsub_cast",
    "backsub_cast_lower",
    "backsub_clip",
    "backsub_clip_lower",
    "backsub_concat",
    "backsub_concat_lower",
    "backsub_conv2d",
    "backsub_conv2d_lower",
    "backsub_convtranspose2d",
    "backsub_convtranspose2d_lower",
    "backsub_cos",
    "backsub_cos_lower",
    "backsub_divcst",
    "backsub_divcst_lower",
    "backsub_div",
    "backsub_div_lower",
    "backsub_dropout",
    "backsub_dropout_lower",
    "backsub_elu",
    "backsub_elu_lower",
    "backsub_exp",
    "backsub_exp_lower",
    "backsub_expand",
    "backsub_expand_lower",
    "backsub_flatten",
    "backsub_flatten_lower",
    "backsub_gather",
    "backsub_gather_lower",
    "backsub_gelu",
    "backsub_gelu_lower",
    "backsub_gemm",
    "backsub_gemm_lower",
    "backsub_leakyrelu",
    "backsub_leakyrelu_lower",
    "backsub_matmul",
    "backsub_matmul_lower",
    "backsub_matmulcst",
    "backsub_matmulcst_lower",
    "backsub_max",
    "backsub_max_lower",
    "backsub_maxpool2d",
    "backsub_maxpool2d_lower",
    "backsub_min",
    "backsub_min_lower",
    "backsub_mulcst",
    "backsub_mulcst_lower",
    "backsub_mul",
    "backsub_mul_lower",
    "backsub_neg",
    "backsub_neg_lower",
    "backsub_pad2d",
    "backsub_pad_lower",
    "backsub_permute",
    "backsub_permute_lower",
    "backsub_pow",
    "backsub_pow_lower",
    "backsub_reciprocal",
    "backsub_reciprocal_lower",
    "backsub_reducemean",
    "backsub_reducemean_lower",
    "backsub_reducesum",
    "backsub_reducesum_lower",
    "backsub_relu",
    "backsub_relu_lower",
    "backsub_reshape",
    "backsub_reshape_lower",
    "backsub_resize",
    "backsub_resize_lower",
    "backsub_scatter",
    "backsub_scatter_lower",
    "backsub_scatternd",
    "backsub_scatternd_lower",
    "backsub_slice",
    "backsub_slice_lower",
    "backsub_shape",
    "backsub_shape_lower",
    "backsub_sin",
    "backsub_sin_lower",
    "backsub_sigmoid",
    "backsub_sigmoid_lower",
    "backsub_softmax",
    "backsub_softmax_lower",
    "backsub_split",
    "backsub_split_lower",
    "backsub_squeeze",
    "backsub_squeeze_lower",
    "backsub_subcst",
    "backsub_subcst_lower",
    "backsub_sub",
    "backsub_sub_lower",
    "backsub_tanh",
    "backsub_tanh_lower",
    "backsub_unsqueeze",
    "backsub_unsqueeze_lower",
    "backsub_upsample",
    "backsub_upsample_lower",
    "backsub_where",
    "backsub_where_lower",
]

from torch import Tensor

from .linear import *
from .._enums import RelaxMode


def _backsub_unary(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if mode == RelaxMode.SINGLE_LINEAR:
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
    if mode == RelaxMode.SINGLE_QUAD:
        raise RuntimeError(
            "Please use `quad_backsub_xxx` for backsubstitution with "
            "quadratic relaxation because the different inputs"
        )
    raise NotImplementedError(f"{mode} is not implemented yet.")


def _backsub_unary_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    if mode == RelaxMode.SINGLE_LINEAR:
        return linear_backsub_unary_lower(
            lower_symbnd_coeffs,
            lower_symbnd_const,
            lower_rlx_coeffs,
            lower_rlx_const,
            upper_rlx_coeffs,
            upper_rlx_const,
        )
    if mode == RelaxMode.SINGLE_QUAD:
        raise RuntimeError(
            "Please use `quad_backsub_xxx_lower` for backsubstitution with "
            "quadratic relaxation because the different inputs"
        )
    raise NotImplementedError(f"{mode} is not implemented yet.")


def backsub_add(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_add(
        lower_symbnd_coeffs, lower_symbnd_const, upper_symbnd_coeffs, upper_symbnd_const
    )


def backsub_add_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_add_lower(lower_symbnd_coeffs, lower_symbnd_const)


def backsub_addcst(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    constant: Tensor,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_addcst(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        constant=constant,
    )


def backsub_addcst_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    constant: Tensor,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return linear_backsub_addcst_lower(
        lower_symbnd_coeffs, lower_symbnd_const, constant=constant
    )


def backsub_argmax(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    dim: int = 0,
    keepdim: bool = False,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    raise NotImplementedError("Backsubstitution for argmax is not implemented yet.")


def backsub_argmax_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    dim: int = 0,
    keepdim: bool = False,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    raise NotImplementedError("Backsubstitution for argmax is not implemented yet.")


def backsub_avgpool2d(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    output_size: tuple[int, int, int],
    kernel_size: tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    output_padding: int | tuple[int, int] = 0,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if lower_symbnd_coeffs.dim() != 4 or upper_symbnd_coeffs.dim() != 4:
        raise NotImplementedError(
            "Backsubstitution of AvgPool only supports 4D tensors (b, c, h, w), "
            f"but got lower_symbnd_coeffs.shape={lower_symbnd_coeffs.shape} and "
            f"upper_symbnd_coeffs.shape={upper_symbnd_coeffs.shape}."
        )

    return linear_backsub_avgpool2d(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        output_size=output_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )


def backsub_avgpool2d_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    output_size: tuple[int, int, int],
    kernel_size: tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    output_padding: int | tuple[int, int] = 0,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    if lower_symbnd_coeffs.dim() != 4:
        raise NotImplementedError(
            "Backsubstitution of AvgPool only supports 4D tensors (b, c, h, w), "
            f"but got lower_symbnd_coeffs.shape={lower_symbnd_coeffs.shape}."
        )

    return linear_backsub_avgpool2d_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        output_size=output_size,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
    )


def backsub_batchnorm2d(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    weight: Tensor,
    bias: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    eps: float,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_batchnorm2d(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        weight=weight,
        bias=bias,
        running_mean=running_mean,
        running_var=running_var,
        eps=eps,
    )


def backsub_batchnorm2d_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    weight: Tensor,
    bias: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    epsilon: float,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return linear_backsub_batchnorm_lower2d(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        weight=weight,
        bias=bias,
        running_mean=running_mean,
        running_var=running_var,
        epsilon=epsilon,
    )


def backsub_cast(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    ori_dtype: str,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_cast(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        ori_dtype=ori_dtype,
    )


def backsub_cast_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    ori_dtype: str,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return linear_backsub_cast_lower(
        lower_symbnd_coeffs, lower_symbnd_const, ori_dtype=ori_dtype
    )


def backsub_clip(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _backsub_unary(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        mode=mode,
    )


def backsub_clip_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return _backsub_unary_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        mode=mode,
    )


def backsub_concat(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    split_size_or_sections: int | list[int],
    dim: int = 0,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
    return linear_backsub_concat(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        split_size_or_sections=split_size_or_sections,
        dim=dim,
    )


def backsub_concat_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    split_size_or_sections: int | list[int],
    dim: int = 0,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[list[Tensor], list[Tensor]]:
    return linear_backsub_concat_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        split_size_or_sections=split_size_or_sections,
        dim=dim,
    )


def backsub_conv2d(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    weight: Tensor,
    bias: Tensor,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    output_padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if mode == RelaxMode.SINGLE_LINEAR:
        return linear_backsub_conv2d(
            lower_symbnd_coeffs,
            lower_symbnd_const,
            upper_symbnd_coeffs,
            upper_symbnd_const,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
        )
    if mode == RelaxMode.SINGLE_QUAD:
        raise RuntimeError(
            "Please use `quad_backsub_conv2d` for backsubstitution with quadratic "
            "relaxation because the different inputs"
        )
    raise NotImplementedError(f"{mode} is not implemented yet.")


def backsub_conv2d_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    weight: Tensor,
    bias: Tensor,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    output_padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    if mode == RelaxMode.SINGLE_LINEAR:
        return linear_backsub_conv2d_lower(
            lower_symbnd_coeffs,
            lower_symbnd_const,
            weight=weight,
            bias=bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
        )
    if mode == RelaxMode.SINGLE_QUAD:
        raise RuntimeError(
            "Please use `quad_backsub_conv2d_lower` for backsubstitution with "
            "quadratic relaxation because the different inputs"
        )
    raise NotImplementedError(f"{mode} is not implemented yet.")


def backsub_convtranspose2d(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    weight: Tensor,
    bias: Tensor,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_convtranspose2d(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def backsub_convtranspose2d_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    weight: Tensor,
    bias: Tensor,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return linear_backsub_convtranspose2d_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def backsub_cos(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _backsub_unary(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        mode=mode,
    )


def backsub_cos_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return _backsub_unary_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        mode=mode,
    )


def backsub_div(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs_a: Tensor,
    lower_rlx_coeffs_b: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs_a: Tensor,
    upper_rlx_coeffs_b: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_div(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        lower_rlx_coeffs_a,
        lower_rlx_coeffs_b,
        lower_rlx_const,
        upper_rlx_coeffs_a,
        upper_rlx_coeffs_b,
        upper_rlx_const,
    )


def backsub_div_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs_a: Tensor,
    lower_rlx_coeffs_b: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs_a: Tensor,
    upper_rlx_coeffs_b: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_div_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs_a,
        lower_rlx_coeffs_b,
        lower_rlx_const,
        upper_rlx_coeffs_a,
        upper_rlx_coeffs_b,
        upper_rlx_const,
    )


def backsub_divcst(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    constant: Tensor,
    is_left_op: bool = True,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_divcst(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        constant=constant,
        is_left_op=is_left_op,
    )


def backsub_divcst_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    constant: Tensor,
    is_left_op: bool = True,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return linear_backsub_divcst_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        constant=constant,
        is_left_op=is_left_op,
    )


def backsub_dropout(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    mask_reserved: Tensor,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_dropout(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        mask_reserved=mask_reserved,
    )


def backsub_dropout_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    mask_reserved: Tensor,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return linear_backsub_dropout_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        mask_reserved=mask_reserved,
    )


def backsub_elu(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _backsub_unary(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        mode=mode,
    )


def backsub_elu_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return _backsub_unary_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        mode=mode,
    )


def backsub_exp(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _backsub_unary(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        mode=mode,
    )


def backsub_exp_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return _backsub_unary_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        mode=mode,
    )


def backsub_expand(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    ori_shape: tuple[int, ...],
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_expand(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        ori_shape=ori_shape,
    )


def backsub_expand_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    ori_shape: tuple[int, ...],
    size: tuple[int, ...],
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return linear_backsub_expand_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        ori_shape=ori_shape,
    )


def backsub_flatten(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    ori_shape: tuple[int, ...],
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_flatten(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        ori_shape=ori_shape,
    )


def backsub_flatten_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    ori_shape: tuple[int, ...],
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return linear_backsub_flatten_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        ori_shape=ori_shape,
    )


def backsub_gather(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    ori_shape: tuple[int, ...],
    dim: int,
    index: Tensor,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_gather(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        ori_shape=ori_shape,
        dim=dim,
        index=index,
    )


def backsub_gather_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    ori_shape: tuple[int, ...],
    dim: int,
    index: Tensor,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return linear_backsub_gather_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        ori_shape=ori_shape,
        dim=dim,
        index=index,
    )


def backsub_gelu(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _backsub_unary(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        mode=mode,
    )


def backsub_gelu_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return _backsub_unary_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        mode=mode,
    )


def backsub_gemm(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    weight: Tensor,
    bias: Tensor,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if mode == RelaxMode.SINGLE_LINEAR:
        return linear_backsub_gemm(
            lower_symbnd_coeffs,
            lower_symbnd_const,
            upper_symbnd_coeffs,
            upper_symbnd_const,
            weight=weight,
            bias=bias,
        )
    if mode == RelaxMode.SINGLE_QUAD:
        raise RuntimeError(
            "Please use `quad_backsub_gemm` for backsubstitution with quadratic "
            "relaxation because the different inputs"
        )
    raise NotImplementedError(f"{mode} is not implemented yet.")


def backsub_gemm_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    weight: Tensor,
    bias: Tensor,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    if mode == RelaxMode.SINGLE_LINEAR:
        return linear_backsub_gemm_lower(
            lower_symbnd_coeffs, lower_symbnd_const, weight=weight, bias=bias
        )
    if mode == RelaxMode.SINGLE_QUAD:
        raise RuntimeError(
            "Please use `quad_backsub_gemm_lower` for backsubstitution with quadratic "
            "relaxation because the different inputs"
        )
    raise NotImplementedError(f"{mode} is not implemented yet.")


def backsub_leakyrelu(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _backsub_unary(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        mode=mode,
    )


def backsub_leakyrelu_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return _backsub_unary_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        mode=mode,
    )


def backsub_matmul(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs_a: Tensor,
    lower_rlx_coeffs_b: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs_a: Tensor,
    upper_rlx_coeffs_b: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_matmul(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        lower_rlx_coeffs_a,
        lower_rlx_coeffs_b,
        lower_rlx_const,
        upper_rlx_coeffs_a,
        upper_rlx_coeffs_b,
        upper_rlx_const,
    )


def backsub_matmul_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs_a: Tensor,
    lower_rlx_coeffs_b: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs_a: Tensor,
    upper_rlx_coeffs_b: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_matmul_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs_a,
        lower_rlx_coeffs_b,
        lower_rlx_const,
        upper_rlx_coeffs_a,
        upper_rlx_coeffs_b,
        upper_rlx_const,
    )


def backsub_matmulcst(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    weight: Tensor,
    is_left_op: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_matmulcst(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        weight=weight,
        is_left_op=is_left_op,
    )


def backsub_matmulcst_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    weight: Tensor,
    is_left_op: bool = True,
) -> tuple[Tensor, Tensor]:
    return linear_backsub_matmulcst_lower(
        lower_symbnd_coeffs, lower_symbnd_const, weight=weight, is_left_op=is_left_op
    )


def backsub_max(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_max(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
    )


def backsub_max_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return linear_backsub_max_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
    )


def backsub_maxpool2d(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    input_size: tuple[int, int, int],
    output_size: tuple[int, int, int],
    kernel_size: tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if mode == RelaxMode.SINGLE_LINEAR:
        return linear_backsub_maxpool2d(
            lower_symbnd_coeffs,
            lower_symbnd_const,
            upper_symbnd_coeffs,
            upper_symbnd_const,
            lower_rlx_coeffs,
            lower_rlx_const,
            upper_rlx_coeffs,
            upper_rlx_const,
            input_size=input_size,
            output_size=output_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
    if mode == RelaxMode.SINGLE_QUAD:
        raise RuntimeError(
            "Please use `quad_backsub_maxpool2d` for backsubstitution with "
            "quadratic relaxation because the different inputs"
        )
    raise NotImplementedError(f"{mode} is not implemented yet.")


def backsub_maxpool2d_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    input_size: tuple[int, int, int],
    output_size: tuple[int, int, int],
    kernel_size: tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    if mode == RelaxMode.SINGLE_LINEAR:
        return linear_backsub_maxpool2d_lower(
            lower_symbnd_coeffs,
            lower_symbnd_const,
            lower_rlx_coeffs,
            lower_rlx_const,
            upper_rlx_coeffs,
            upper_rlx_const,
            input_size=input_size,
            output_size=output_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
    if mode == RelaxMode.SINGLE_QUAD:
        raise RuntimeError(
            "Please use `quad_backsub_maxpool2d_lower` for backsubstitution with "
            "quadratic relaxation because the different inputs"
        )
    raise NotImplementedError(f"{mode} is not implemented yet.")


def backsub_min(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_min(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
    )


def backsub_min_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return linear_backsub_min_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
    )


def backsub_mul(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs_a: Tensor,
    lower_rlx_coeffs_b: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs_a: Tensor,
    upper_rlx_coeffs_b: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_mul(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        lower_rlx_coeffs_a,
        lower_rlx_coeffs_b,
        lower_rlx_const,
        upper_rlx_coeffs_a,
        upper_rlx_coeffs_b,
        upper_rlx_const,
    )


def backsub_mul_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs_a: Tensor,
    lower_rlx_coeffs_b: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs_a: Tensor,
    upper_rlx_coeffs_b: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_mul_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs_a,
        lower_rlx_coeffs_b,
        lower_rlx_const,
        upper_rlx_coeffs_a,
        upper_rlx_coeffs_b,
        upper_rlx_const,
    )


def backsub_mulcst(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    constant: Tensor,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_mulcst(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        constant=constant,
    )


def backsub_mulcst_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    constant: Tensor,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return linear_backsub_mulcst_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        constant=constant,
    )


def backsub_neg(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_neg(
        lower_symbnd_coeffs, lower_symbnd_const, upper_symbnd_coeffs, upper_symbnd_const
    )


def backsub_neg_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return linear_backsub_neg_lower(lower_symbnd_coeffs, lower_symbnd_const)


def backsub_pad2d(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    pad: tuple[int, int, int, int],
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if lower_symbnd_coeffs.dim() != 4 or upper_symbnd_coeffs.dim() != 4:
        raise NotImplementedError(
            "Backsubstitution of Pad only supports 4D tensors (b, c, h, w), "
            f"but got lower_symbnd_coeffs.shape={lower_symbnd_coeffs.shape} and "
            f"upper_symbnd_coeffs.shape={upper_symbnd_coeffs.shape}."
        )

    return linear_backsub_pad2d(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        pad=pad,
    )


def backsub_pad_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    pad: tuple[int, ...],
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    if lower_symbnd_coeffs.dim() != 4:
        raise NotImplementedError(
            "Backsubstitution of Pad only supports 4D tensors (b, c, h, w), "
            f"but got lower_symbnd_coeffs.shape={lower_symbnd_coeffs.shape}."
        )

    return linear_backsub_pad2d_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        pad=pad,
    )


def backsub_permute(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    dims: tuple[int, ...],
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_permute(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        dims=dims,
    )


def backsub_permute_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    dims: tuple[int, ...],
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return linear_backsub_permute_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        dims=dims,
    )


def backsub_pow(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _backsub_unary(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        mode=mode,
    )


def backsub_pow_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return _backsub_unary_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        mode=mode,
    )


def backsub_reciprocal(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _backsub_unary(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        mode=mode,
    )


def backsub_reciprocal_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return _backsub_unary_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        mode=mode,
    )


def backsub_reducemean(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    ori_shape: tuple[int, ...],
    dim: int = 0,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_reducemean(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        ori_shape=ori_shape,
        dim=dim,
    )


def backsub_reducemean_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    ori_shape: tuple[int, ...],
    dim: int = 0,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return linear_backsub_reducemean_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        ori_shape,
        dim=dim,
    )


def backsub_reducesum(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    ori_shape: tuple[int, ...],
    dim: int = 0,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_reducesum(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        ori_shape=ori_shape,
        dim=dim,
    )


def backsub_reducesum_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    ori_shape: tuple[int, ...],
    dim: int = 0,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return linear_backsub_reducesum_lower(
        lower_symbnd_coeffs, lower_symbnd_const, ori_shape=ori_shape, dim=dim
    )


def backsub_relu(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _backsub_unary(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        mode=mode,
    )


def backsub_relu_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return _backsub_unary_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        mode=mode,
    )


def backsub_reshape(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    ori_shape: tuple[int, ...],
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_reshape(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        ori_shape=ori_shape,
    )


def backsub_reshape_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    ori_shape: tuple[int, ...],
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return linear_backsub_reshape_lower(
        lower_symbnd_coeffs, lower_symbnd_const, ori_shape=ori_shape
    )


def backsub_resize(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    input_size: tuple[int, ...],
    scale_factor: tuple[float, ...],
    resize_mode: str = "nearest",
    align_corners: bool = False,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_resize(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        input_size=input_size,
        scale_factor=scale_factor,
        mode=resize_mode,
        align_corners=align_corners,
    )


def backsub_resize_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    input_size: tuple[int, ...],
    scale_factor: tuple[float, ...],
    resize_mode: str = "nearest",
    align_corners: bool = False,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return linear_backsub_resize_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        input_size=input_size,
        scale_factor=scale_factor,
        mode=resize_mode,
        align_corners=align_corners,
    )


def backsub_scatter(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    dim: int,
    index: Tensor,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_scatter(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        dim=dim,
        index=index,
    )


def backsub_scatter_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    dim: int,
    index: Tensor,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_scatter_lower(
        lower_symbnd_coeffs, lower_symbnd_const, dim=dim, index=index
    )


def backsub_scatternd(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    index: Tensor,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_scatternd(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        index=index,
    )


def backsub_scatternd_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    index: Tensor,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_scatternd_lower(
        lower_symbnd_coeffs, lower_symbnd_const, index=index
    )


def backsub_slice(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    ori_shape: tuple[int, ...],
    starts: list[int],
    ends: list[int],
    axes: list[int],
    steps: list[int],
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_slice(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        ori_shape=ori_shape,
        starts=starts,
        ends=ends,
        axes=axes,
        steps=steps,
    )


def backsub_slice_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    ori_shape: tuple[int, ...],
    starts: list[int],
    ends: list[int],
    axes: list[int],
    steps: list[int],
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return linear_backsub_slice_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        ori_shape=ori_shape,
        starts=starts,
        ends=ends,
        axes=axes,
        steps=steps,
    )


def backsub_shape(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    raise NotImplementedError("Backsubstitution for shape is not implemented yet.")


def backsub_shape_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    raise NotImplementedError("Backsubstitution for shape is not implemented yet.")


def backsub_sigmoid(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _backsub_unary(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        mode=mode,
    )


def backsub_sigmoid_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return _backsub_unary_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        mode=mode,
    )


def backsub_sin(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _backsub_unary(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        mode=mode,
    )


def backsub_sin_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return _backsub_unary_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        mode=mode,
    )


def backsub_softmax(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    dim: int = -1,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_softmax(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        dim=dim,
    )


def backsub_softmax_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    dim: int = -1,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return linear_backsub_softmax_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        dim=dim,
    )


def backsub_split(
    lower_symbnd_coeffs_list: list[Tensor],
    lower_symbnd_const_list: list[Tensor],
    upper_symbnd_coeffs_list: list[Tensor],
    upper_symbnd_const_list: list[Tensor],
    *,
    dim: int = 0,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_split(
        lower_symbnd_coeffs_list,
        lower_symbnd_const_list,
        upper_symbnd_coeffs_list,
        upper_symbnd_const_list,
        dim=dim,
    )


def backsub_split_lower(
    lower_symbnd_coeffs_list: list[Tensor],
    lower_symbnd_const_list: list[Tensor],
    *,
    dim: int = 0,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return linear_backsub_split_lower(
        lower_symbnd_coeffs_list,
        lower_symbnd_const_list,
        dim=dim,
    )


def backsub_squeeze(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    dim: int | None = None,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_squeeze(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        dim=dim,
    )


def backsub_squeeze_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    dim: int | None = None,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return linear_backsub_squeeze_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        dim=dim,
    )


def backsub_sub(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_sub(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
    )


def backsub_sub_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_sub_lower(lower_symbnd_coeffs, lower_symbnd_const)


def backsub_subcst(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    constant: Tensor,
    is_left_op: bool = True,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_subcst(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        constant=constant,
        is_left_op=is_left_op,
    )


def backsub_subcst_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    constant: Tensor,
    is_left_op: bool = True,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return linear_backsub_subcst_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        constant=constant,
        is_left_op=is_left_op,
    )


def backsub_tanh(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return _backsub_unary(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        mode=mode,
    )


def backsub_tanh_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_rlx_coeffs: Tensor,
    lower_rlx_const: Tensor,
    upper_rlx_coeffs: Tensor,
    upper_rlx_const: Tensor,
    *,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return _backsub_unary_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_rlx_coeffs,
        lower_rlx_const,
        upper_rlx_coeffs,
        upper_rlx_const,
        mode=mode,
    )


def backsub_unsqueeze(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    dim: int,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_unsqueeze(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        dim=dim,
    )


def backsub_unsqueeze_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    dim: int,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return linear_backsub_unsqueeze_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        dim=dim,
    )


def backsub_upsample(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    scale_factor: tuple[float, ...],
    upsample_mode: str = "nearest",
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_upsample(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        scale_factor=scale_factor,
        mode=upsample_mode,
    )


def backsub_upsample_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    scale_factor: tuple[float, ...],
    upsample_mode: str = "nearest",
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor]:
    return linear_backsub_upsample_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        scale_factor=scale_factor,
        mode=upsample_mode,
    )


def backsub_where(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    mask: Tensor,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_where(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
        mask=mask,
    )


def backsub_where_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    mask: Tensor,
    mode: RelaxMode = RelaxMode.SINGLE_LINEAR,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return linear_backsub_where_lower(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        mask=mask,
    )
