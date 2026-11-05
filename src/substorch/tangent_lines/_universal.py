__all__ = ["cal_tangent_line"]

from functools import partial
from typing import Callable, Tuple

import torch
from torch import Tensor

from ._exception import NotConverged


def _obj_error(
    x: Tensor,
    fixed_x: Tensor,
    fixed_y: Tensor,
    f: Callable,
    df: Callable,
) -> Tensor:
    cur_y = df(x) * (fixed_x - x) + f(x)
    return cur_y - fixed_y


def _get_interval_by_sampling(
    l: Tensor,
    r: Tensor,
    obj_error_p: Callable,
    n_samples: int = 10,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    device = l.device
    n = l.shape[0]
    samples = torch.linspace(0, 1, n_samples, device=device)
    samples = samples.unsqueeze(-1)
    samples = samples * (r - l) + l

    samples_error = obj_error_p(samples)
    sign_changes = (samples_error[:-1] * samples_error[1:]) <= 0

    if not torch.all(torch.any(sign_changes, dim=0)):
        raise ValueError(f"All points in [{l}, {r}] have the same sign of the error. ")

    idx = sign_changes.float().argmax(dim=0)

    c_idxs = list(range(sign_changes.size(1)))
    l_idx = idx
    r_idx = torch.minimum(idx + 1, torch.tensor(sign_changes.size(0)))

    l = samples[l_idx, c_idxs].unsqueeze(0)
    r = samples[r_idx, c_idxs].unsqueeze(0)
    l_error = samples_error[l_idx, c_idxs].unsqueeze(0)
    r_error = samples_error[r_idx, c_idxs].unsqueeze(0)

    return l, r, l_error, r_error


def cal_tangent_line(
    fixed_x: Tensor,
    another_x: Tensor,
    f: Callable[[Tensor], Tensor],
    df: Callable[[Tensor], Tensor],
    max_iter: int = 100,
    num_samples: int = 20,
    tol: float = 1e-5,
) -> Tuple[Tensor, Tensor]:
    """Find the point x1 such that the tangent line of f at x1 passes through the fixed point x0."""
    assert fixed_x.dim() == 1
    assert fixed_x.shape == another_x.shape
    n = fixed_x.shape[0]
    device = fixed_x.device

    fixed_x = fixed_x.unsqueeze(0)
    another_x = another_x.unsqueeze(0)

    mask = fixed_x < another_x
    left = torch.where(mask, fixed_x, another_x)
    right = torch.where(mask, another_x, fixed_x)

    obj_error_p = partial(_obj_error, fixed_x=fixed_x, fixed_y=f(fixed_x), f=f, df=df)
    left, right, error_left, error_right = _get_interval_by_sampling(
        left, right, obj_error_p, num_samples
    )

    active = torch.ones((1, n), dtype=torch.bool, device=device)
    for _ in range(max_iter):
        mid = (left + right) * 0.5
        error_mid = obj_error_p(mid)
        active &= error_mid.abs() >= tol

        if not active.any():
            x1 = (left + right) / 2
            slope = df(x1)
            return slope, x1

        same_sign = (error_left * error_mid) > 0

        left = torch.where(same_sign & active, mid, left)
        error_left = torch.where(same_sign & active, error_mid, error_left)
        right = torch.where((~same_sign) & active, mid, right)
        error_right = torch.where((~same_sign) & active, error_mid, error_right)

    raise NotConverged(f"\nfixed_x={fixed_x}\nanother_x={another_x}\nf={f}")
