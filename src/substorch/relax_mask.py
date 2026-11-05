__all__ = [
    "get_rlx_mask_relu",
    "get_rlx_mask_sigmoid",
    "get_rlx_mask_tanh",
    "get_rlx_mask_elu",
    "get_rlx_mask_leakyrelu",
    "get_rlx_mask_silu",
    "get_rlx_mask_maxpool2d",
]

from torch import Tensor


def get_rlx_mask_relu(
    pre_lower_bound: Tensor,
    pre_upper_bound: Tensor,
    lower_bound: Tensor,
    upper_bound: Tensor,
    *,
    min_range: float,
) -> Tensor:
    r_min = min_range
    r_min_half = r_min * 0.5
    r = -pre_lower_bound * pre_upper_bound
    return (
        (pre_lower_bound < -r_min_half) & (pre_upper_bound > r_min_half) & (r > r_min)
    )


def get_rlx_mask_sigmoid(
    pre_lower_bound: Tensor,
    pre_upper_bound: Tensor,
    lower_bound: Tensor,
    upper_bound: Tensor,
    *,
    min_range: float,
    sigmoid_relax_limit: float,
) -> Tensor:
    r_min = min_range
    limit = sigmoid_relax_limit
    r = (upper_bound - lower_bound) * (pre_upper_bound - pre_lower_bound)
    return (pre_lower_bound < limit) & (pre_upper_bound > -limit) & (r > r_min)


def get_rlx_mask_tanh(
    pre_lower_bound: Tensor,
    pre_upper_bound: Tensor,
    lower_bound: Tensor,
    upper_bound: Tensor,
    *,
    min_range: float,
    tanh_relax_limit: float,
) -> Tensor:
    r_min = min_range
    limit = tanh_relax_limit
    r = (upper_bound - lower_bound) * (pre_upper_bound - pre_lower_bound)
    return (pre_lower_bound < limit) & (pre_upper_bound > -limit) & (r > r_min)


def get_rlx_mask_elu(
    pre_lower_bound: Tensor,
    pre_upper_bound: Tensor,
    lower_bound: Tensor,
    upper_bound: Tensor,
    *,
    min_range: float,
) -> Tensor:
    r_min = min_range
    r_min_half = r_min * 0.5
    r = (upper_bound - lower_bound) * (pre_upper_bound - pre_lower_bound)
    return (
        (pre_lower_bound < -r_min_half) & (pre_upper_bound > r_min_half) & (r > r_min)
    )


def get_rlx_mask_leakyrelu(
    pre_lower_bound: Tensor,
    pre_upper_bound: Tensor,
    lower_bound: Tensor,
    upper_bound: Tensor,
    *,
    min_range: float,
) -> Tensor:
    r_min = min_range
    r_min_half = r_min * 0.5
    r = -pre_lower_bound * pre_upper_bound
    return (
        (pre_lower_bound < -r_min_half) & (pre_upper_bound > r_min_half) & (r > r_min)
    )


def get_rlx_mask_silu(
    pre_lower_bound: Tensor,
    pre_upper_bound: Tensor,
    lower_bound: Tensor,
    upper_bound: Tensor,
    *,
    min_range: float,
) -> Tensor:
    r_min = min_range
    r_min_half = r_min * 0.5
    return (
        (pre_lower_bound < -r_min_half)
        & (pre_upper_bound > r_min_half)
        & ((upper_bound - lower_bound) * (pre_upper_bound - pre_lower_bound) > r_min)
    )


def get_rlx_mask_maxpool2d(
    pre_lower_bound: Tensor,
    pre_upper_bound: Tensor,
    pre_lower_bound_max: Tensor,
    pre_lower_bound_argmax: Tensor,
) -> Tensor:
    return pre_lower_bound_max < pre_upper_bound.topk(2, dim=1).values[:, 1]
