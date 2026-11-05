__all__ = ["find_extreme_point_of_linear_symbnds"]

import torch
from torch import Tensor


def find_extreme_point_of_linear_symbnds(
    symbnd_coeffs: Tensor,
    lower_bounds: Tensor,
    upper_bounds: Tensor,
    is_lower: bool = True,
) -> Tensor:
    lower_bounds = lower_bounds.unsqueeze(0)
    upper_bounds = upper_bounds.unsqueeze(0)
    if is_lower:
        return torch.where(symbnd_coeffs >= 0, lower_bounds, upper_bounds)
    return torch.where(symbnd_coeffs >= 0, upper_bounds, lower_bounds)
