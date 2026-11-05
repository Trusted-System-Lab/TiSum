__all__ = ["batched_cal_quad_scalar_bound", "batched_cal_quad_scalar_bound_lower"]

import torch
from torch import Tensor


def _batched_cal_quad_scalar_bound(
    symbnd_quad_coeffs: Tensor,
    symbnd_linear_coeffs: Tensor,
    symbnd_const: Tensor,
    sqr_scalar_bound1: Tensor,
    sqr_scalar_bound2: Tensor,
    scalar_bound1: Tensor,
    scalar_bound2: Tensor,
) -> Tensor:
    pos_symbnd_quad_coeffs = symbnd_quad_coeffs.clamp_min(0)
    neg_symbnd_quad_coeffs = symbnd_quad_coeffs.clamp_max(0)
    pos_symbnd_linear_coeffs = symbnd_linear_coeffs.clamp_min(0)
    neg_symbnd_linear_coeffs = symbnd_linear_coeffs.clamp_max(0)

    d = list(range(2, len(symbnd_linear_coeffs.shape)))
    scalar_bound = (
        symbnd_const
        + (pos_symbnd_linear_coeffs * scalar_bound1).sum(dim=d)
        + (neg_symbnd_linear_coeffs * scalar_bound2).sum(dim=d)
        + (pos_symbnd_quad_coeffs * sqr_scalar_bound1).sum(dim=d)
        + (neg_symbnd_quad_coeffs * sqr_scalar_bound2).sum(dim=d)
    )

    return scalar_bound


def batched_cal_quad_scalar_bound(
    lower_symbnd_quad_coeffs: Tensor,
    lower_symbnd_linear_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_quad_coeffs: Tensor,
    upper_symbnd_linear_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_scalar_bounds: Tensor,
    upper_scalar_bounds: Tensor,
) -> tuple[Tensor, Tensor]:
    lower_scalar_bounds = lower_scalar_bounds.unsqueeze(1)
    upper_scalar_bounds = upper_scalar_bounds.unsqueeze(1)

    sqr_lower_scalar_bounds = lower_scalar_bounds**2
    sqr_upper_scalar_bounds = upper_scalar_bounds**2
    quad_lower_scalar_bounds = torch.minimum(
        sqr_lower_scalar_bounds, sqr_upper_scalar_bounds
    )
    quad_upper_scalar_bounds = torch.maximum(
        sqr_lower_scalar_bounds, sqr_upper_scalar_bounds
    )

    new_lower_scalar_bound = _batched_cal_quad_scalar_bound(
        lower_symbnd_quad_coeffs,
        lower_symbnd_linear_coeffs,
        lower_symbnd_const,
        quad_lower_scalar_bounds,
        quad_upper_scalar_bounds,
        lower_scalar_bounds,
        upper_scalar_bounds,
    )
    new_upper_scalar_bound = _batched_cal_quad_scalar_bound(
        upper_symbnd_quad_coeffs,
        upper_symbnd_linear_coeffs,
        upper_symbnd_const,
        quad_upper_scalar_bounds,
        quad_lower_scalar_bounds,
        upper_scalar_bounds,
        lower_scalar_bounds,
    )

    return new_lower_scalar_bound, new_upper_scalar_bound


def batched_cal_quad_scalar_bound_lower(
    lower_symbnd_quad_coeffs: Tensor,
    lower_symbnd_linear_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_scalar_bounds: Tensor,
    upper_scalar_bounds: Tensor,
) -> Tensor:
    lower_scalar_bounds = lower_scalar_bounds.unsqueeze(1)
    upper_scalar_bounds = upper_scalar_bounds.unsqueeze(1)

    sqr_lower_scalar_bounds = lower_scalar_bounds**2
    sqr_upper_scalar_bounds = upper_scalar_bounds**2
    quad_lower_scalar_bounds = torch.minimum(
        sqr_lower_scalar_bounds, sqr_upper_scalar_bounds
    )
    quad_upper_scalar_bounds = torch.maximum(
        sqr_lower_scalar_bounds, sqr_upper_scalar_bounds
    )

    return _batched_cal_quad_scalar_bound(
        lower_symbnd_quad_coeffs,
        lower_symbnd_linear_coeffs,
        lower_symbnd_const,
        quad_lower_scalar_bounds,
        quad_upper_scalar_bounds,
        lower_scalar_bounds,
        upper_scalar_bounds,
    )
