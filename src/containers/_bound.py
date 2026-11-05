__all__ = ["Bound", "SparseBound"]

from dataclasses import dataclass

import torch
from torch import Tensor

_TOLERANCE = 1e-6


@dataclass(slots=True)
class Bound:
    """A class of two tensors representing lower and upper scalar bounds."""

    l: Tensor
    u: Tensor | None

    def __post_init__(self):
        if self.u is not None:
            assert self.l.shape == self.u.shape
            assert torch.all(self.l <= self.u - _TOLERANCE), (
                "The lower bound must be less than the upper bound. "
                "but \n"
                f"l={self.l[self.l > self.u].tolist()}\n"
                f"u={self.u[self.l > self.u].tolist()}\n"
            )

    def intersect(self, other: "Bound") -> "Bound":
        """This function computes the intersection of two scalar bounds as intervals."""
        return Bound(
            l=torch.maximum(self.l, other.l),
            u=(
                torch.minimum(self.u, other.u)
                if self.u is not None and other.u is not None
                else None
            ),
        )

    def __add__(self, other: "Bound") -> "Bound":
        return Bound(
            l=self.l + other.l,
            u=None if self.u is None or other.u is None else self.u + other.u,
        )

    def __sub__(self, other: "Bound") -> "Bound":
        return Bound(
            l=self.l - other.l,
            u=None if self.u is None or other.u is None else self.u - other.u,
        )

    @property
    def shape(self):
        return self.l.shape

    def reshape(self, *shape: int) -> "Bound":
        """This function reshapes the scalar bounds."""
        return Bound(
            l=self.l.reshape(*shape),
            u=None if self.u is None else self.u.reshape(*shape),
        )

    def to(self, *args, **kwargs) -> "Bound":
        """This is a similar function to the `torch.Tensor.to()` function."""
        return Bound(
            l=self.l.to(*args, **kwargs),
            u=None if self.u is None else self.u.to(*args, **kwargs),
        )

    def clone(self) -> "Bound":
        """This is a similar function to the `torch.Tensor.clone()` function."""
        return Bound(
            l=self.l.clone(),
            u=None if self.u is None else self.u.clone(),
        )

    def detach(self) -> "Bound":
        """This is a similar function to the `torch.Tensor.detach()` function."""
        return Bound(
            l=self.l.detach(),
            u=None if self.u is None else self.u.detach(),
        )

    def numel(self) -> int:
        """This function returns the number of elements in the scalar bounds."""
        return self.l.numel()

    def size(self, dim: int) -> int:
        """This function returns the size of the scalar bounds."""
        return self.l.size(dim)

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"l.shape={self.l.shape}, "
            f"u.shape={self.u.shape if self.u is not None else None}"
            f")"
        )

    def __repr__(self):
        return self.__str__()


@dataclass(slots=True)
class SparseBound(Bound):
    """A class of two tensors representing lower and upper scalar bounds with sparse indices."""

    ori_shape: tuple[int, ...]
    indices: Tensor



    def detach(self) -> "SparseBound":
        return SparseBound(
            ori_shape=tuple(s for s in self.ori_shape),
            indices=self.indices.clone().detach(),
            l=self.l.detach(),
            u=self.u.detach() if self.u is not None else None,
        )

    def clone(self) -> "SparseBound":
        return SparseBound(
            ori_shape=tuple(s for s in self.ori_shape),
            indices=self.indices.clone(),
            l=self.l.clone(),
            u=self.u.clone() if self.u is not None else None,
        )

    def to(self, *args, **kwargs) -> "SparseBound":
        device = kwargs.get("device", self.indices.device)
        return SparseBound(
            ori_shape=tuple(s for s in self.ori_shape),
            indices=self.indices.to(device=device),
            l=self.l.to(*args, **kwargs),
            u=self.u.to(*args, **kwargs) if self.u is not None else None,
        )

    def reshape(self, *shape: int) -> "SparseBound":
        return SparseBound(
            ori_shape=shape,
            indices=self.indices.clone(),
            l=self.l.reshape(*shape),
            u=None if self.u is None else self.u.reshape(*shape),
        )

    def intersect(self, other: "SparseBound") -> "SparseBound":
        assert self.ori_shape == other.ori_shape
        assert self.indices.shape == other.indices.shape
        assert torch.all(self.indices == other.indices)

        return SparseBound(
            ori_shape=self.ori_shape,
            indices=self.indices,
            l=torch.maximum(self.l, other.l),
            u=(
                torch.minimum(self.u, other.u)
                if self.u is not None and other.u is not None
                else None
            ),
        )

    def resparse(self, indices: Tensor) -> "SparseBound":
        assert indices.numel() <= self.indices.numel()
        if indices.numel() == self.indices.numel():
            assert torch.all(indices == indices.indices)
            return self

        dtype, device = self.l.dtype, self.l.device
        new_mask = torch.zeros(self.ori_shape, dtype=torch.bool, device=device)
        new_mask[indices] = True
        new_mask = new_mask[self.indices]
        new_l = self.l[new_mask]
        new_u = self.u[new_mask] if self.u is not None else None
        return SparseBound(
            ori_shape=self.ori_shape,
            indices=indices,
            l=new_l,
            u=new_u,
        )

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"shape={self.ori_shape}, "
            f"indices.shape={self.indices.shape}, "
            f"l.shape={self.l.shape}, "
            f"u.shape={self.u.shape if self.u is not None else None}"
            f")"
        )

    def __repr__(self):
        return self.__str__()
