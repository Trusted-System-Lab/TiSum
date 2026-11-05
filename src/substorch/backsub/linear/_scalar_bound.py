__all__ = ["cal_scalar_bound", "cal_scalar_bound_lower"]

from torch import Tensor


def _cal_scalar_bound(
    symbnd_coeffs: Tensor,
    symbnd_const: Tensor,
    scalar_bound1: Tensor,
    scalar_bound2: Tensor,
) -> Tensor:
    pos_symbnd_coeffs = symbnd_coeffs.clamp_min(0)
    neg_symbnd_coeffs = symbnd_coeffs.clamp_max(0)

    d = list(range(1, len(symbnd_coeffs.shape)))
    scalar_bound = (
        symbnd_const
        + (pos_symbnd_coeffs * scalar_bound1).sum(dim=d)
        + (neg_symbnd_coeffs * scalar_bound2).sum(dim=d)
    )

    return scalar_bound


def cal_scalar_bound(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    upper_symbnd_coeffs: Tensor,
    upper_symbnd_const: Tensor,
    lower_scalar_bounds: Tensor,
    upper_scalar_bounds: Tensor,
) -> tuple[Tensor, Tensor]:
    lower_scalar_bounds = lower_scalar_bounds.unsqueeze(0)
    upper_scalar_bounds = upper_scalar_bounds.unsqueeze(0)

    new_lower_scalar_bound = _cal_scalar_bound(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_scalar_bounds,
        upper_scalar_bounds,
    )
    new_upper_scalar_bound = _cal_scalar_bound(
        upper_symbnd_coeffs,
        upper_symbnd_const,
        upper_scalar_bounds,
        lower_scalar_bounds,
    )

    return new_lower_scalar_bound, new_upper_scalar_bound


def cal_scalar_bound_lower(
    lower_symbnd_coeffs: Tensor,
    lower_symbnd_const: Tensor,
    lower_scalar_bounds: Tensor,
    upper_scalar_bounds: Tensor,
) -> Tensor:
    lower_scalar_bounds = lower_scalar_bounds.unsqueeze(0)
    upper_scalar_bounds = upper_scalar_bounds.unsqueeze(0)

    return _cal_scalar_bound(
        lower_symbnd_coeffs,
        lower_symbnd_const,
        lower_scalar_bounds,
        upper_scalar_bounds,
    )
