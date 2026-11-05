__all__ = [
    "batched_linear_backsub_flatten",
    "batched_linear_backsub_flatten_lower",
]


from torch import Tensor


def batched_linear_backsub_flatten(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    *,
    ori_shape: tuple[int, ...],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    t, b = lower_symbnd_coeffs.shape[:2]
    lower_symbnd_coeffs = lower_symbnd_coeffs.reshape(t, b, *ori_shape)
    upper_symbnd_coeffs = upper_symbnd_coeffs.reshape(t, b, *ori_shape)

    return (
        lower_symbnd_coeffs,
        lower_symbnd_const,
        upper_symbnd_coeffs,
        upper_symbnd_const,
    )


def batched_linear_backsub_flatten_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    *,
    ori_shape: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    t, b = lower_symbnd_coeffs.shape[:2]
    lower_symbnd_coeffs = lower_symbnd_coeffs.reshape(t, b, *ori_shape)

    return lower_symbnd_coeffs, lower_symbnd_const
