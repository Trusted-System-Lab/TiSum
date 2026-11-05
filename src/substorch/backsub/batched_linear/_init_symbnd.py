__all__ = [
    "batched_init_symbnd_similar_eye",
    "batched_init_symbnd_gemm",
    "batched_init_symbnd_matmulcst",
]


import torch
from torch import Tensor

from ..linear import init_symbnd_similar_eye, init_symbnd_gemm


def batched_init_symbnd_similar_eye(
    output_size: tuple[int, ...],
    *,
    n_batches: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    symbnd_coeffs, symbnd_const = init_symbnd_similar_eye(
        output_size, dtype=dtype, device=device
    )
    d = len(output_size) + 1
    symbnd_coeffs = symbnd_coeffs.unsqueeze(0).repeat(n_batches, *([1] * d))
    symbnd_const = symbnd_const.unsqueeze(0).repeat(n_batches, 1)

    return symbnd_coeffs.detach(), symbnd_const.detach()


def batched_init_symbnd_gemm(
    weight: Tensor,
    bias: Tensor,
    *,
    n_batches: int,
) -> tuple[Tensor, Tensor]:
    symbnd_coeffs, symbnd_const = init_symbnd_gemm(weight, bias)
    d = len(symbnd_coeffs.shape)
    symbnd_coeffs = symbnd_coeffs.unsqueeze(0).repeat(n_batches, *([1] * d))
    symbnd_const = symbnd_const.unsqueeze(0).repeat(n_batches, 1)

    return symbnd_coeffs.detach_(), symbnd_const.detach_()


def batched_init_symbnd_matmulcst(
    weight: Tensor,
    *,
    n_batches: int,
) -> tuple[Tensor, Tensor]:
    symbnd_coeffs = weight.t().clone()
    symbnd_const = torch.zeros(
        weight.shape[1], dtype=weight.dtype, device=weight.device
    )
    d = len(symbnd_coeffs.shape)
    symbnd_coeffs = symbnd_coeffs.unsqueeze(0).repeat(n_batches, *([1] * d))
    symbnd_const = symbnd_const.unsqueeze(0).repeat(n_batches, 1)

    return symbnd_coeffs.detach_(), symbnd_const.detach_()
