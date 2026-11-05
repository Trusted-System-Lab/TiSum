__all__ = ["SparseParam"]

from dataclasses import dataclass

import torch
from torch import Tensor

_TOLERANCE = 1e-6


@dataclass(slots=True)
class SparseParam:
    """A data structure contain the parameter tensor P for the optimized neurons."""

    param: Tensor
    indices: Tensor
    mask: Tensor

    def __post_init__(self):
        assert self.param.requires_grad
        assert self.indices.dtype == torch.long
        assert self.mask.dtype == torch.bool
        assert self.param.dim() == 3
        assert self.indices.dim() == 1
        assert self.mask.dim() == 1
        assert self.param.shape[0] == self.indices.shape[0]

    def to(self, *args, **kwargs) -> "SparseParam":
        """This is a similar function to the `torch.Tensor.to()` function."""

        dtype = kwargs.pop("dtype", self.param.dtype)
        return SparseParam(
            param=self.param.to(*args, **kwargs, dtype=dtype),
            indices=self.indices.to(*args, dtype=torch.long, **kwargs),
            mask=self.mask.to(*args, dtype=torch.long, **kwargs),
        )

    def clone(self) -> "SparseParam":
        """This is a similar function to the `torch.Tensor.clone()` function."""
        return SparseParam(
            param=self.param.clone(),
            indices=self.indices.clone(),
            mask=self.mask.clone(),
        )

    def detach(self) -> "SparseParam":
        """This is a similar function to the `torch.Tensor.detach()` function."""
        return SparseParam(
            param=self.param.detach(),
            indices=self.indices.detach(),
            mask=self.mask.detach(),
        )

    def detach_(self) -> "SparseParam":
        """This is a similar function to the `torch.Tensor.detach_()` function."""
        self.param.detach_()
        self.indices.detach_()
        self.mask.detach_()
        return self

    def requires_grad_(self, requires_grad: bool) -> "SparseParam":
        """This is a similar function to the `torch.Tensor.requires_grad_()`
        function."""
        self.param.requires_grad_(requires_grad)
        return self

    def __str__(self):
        return (
            f"SparseParam("
            f"param.shape={self.param.shape}, "
            f"indices.shape={self.indices.shape}, "
            f"mask.shape={self.mask.shape}"
            f")"
        )

    def __repr__(self):
        return self.__str__()
