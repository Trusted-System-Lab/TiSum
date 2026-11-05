__all__ = ["batched_interval_gemm"]

from torch import Tensor


def batched_interval_gemm(
    x_l: Tensor,
    x_u: Tensor,
    *,
    weight: Tensor,
    bias: Tensor,
) -> tuple[Tensor, Tensor]:
    w = weight
    b = bias

    w_p = w.clamp_min(0)
    w_n = w.clamp_max(0)
    y_l = b + x_l @ w_p + x_u @ w_n
    y_u = b + x_u @ w_p + x_l @ w_n

    return y_l, y_u
