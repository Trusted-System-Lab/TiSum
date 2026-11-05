__all__ = ["batched_interval_matmulcst"]

from torch import Tensor


def batched_interval_matmulcst(
    x_l: Tensor,
    x_u: Tensor,
    *,
    weight: Tensor,
    is_left_op: bool = True,
) -> tuple[Tensor, Tensor]:
    w_p, w_n = weight.clamp_min(0), weight.clamp_max(0)

    if is_left_op:
        y_l = w_p @ x_l + w_n @ x_u
        y_u = w_p @ x_u + w_n @ x_l
    else:
        y_l = x_l @ w_p + x_u @ w_n
        y_u = x_u @ w_p + x_l @ w_n

    return y_l, y_u
