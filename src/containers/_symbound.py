__all__ = ["LinSym", "SparseLinSym", "QuadSym", "SymBound"]

from dataclasses import dataclass

import torch
from torch import Tensor

_TOLERANCE = 1e-6


@dataclass(slots=True)
class LinSym:
    """A class of two tensors representing linear constraints."""

    A: Tensor
    b: Tensor

    def __post_init__(self):
        assert self.A.shape[0] == self.b.shape[0]

    def to(self, *args, **kwargs) -> "LinSym":
        """This is a similar function to the `torch.Tensor.to()` function."""
        return LinSym(A=self.A.to(*args, **kwargs), b=self.b.to(*args, **kwargs))

    def clone(self) -> "LinSym":
        """This is a similar function to the `torch.Tensor.clone()` function."""
        return LinSym(A=self.A.clone(), b=self.b.clone())

    def detach(self) -> "LinSym":
        """This is a similar function to the `torch.Tensor.detach()` function."""
        return LinSym(A=self.A.detach(), b=self.b.detach())

    def __add__(self, other: "LinSym") -> "LinSym":
        return LinSym(A=self.A + other.A, b=self.b + other.b)

    def __sub__(self, other: "LinSym") -> "LinSym":
        return LinSym(A=self.A - other.A, b=self.b - other.b)

    def __iadd__(self, other: "LinSym"):
        self.A += other.A
        self.b += other.b
        return self

    def __isub__(self, other: "LinSym"):
        self.A -= other.A
        self.b -= other.b
        return self

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"A.shape={self.A.shape}, "
            f"b={self.b.shape}"
            f")"
        )

    def __repr__(self):
        return self.__str__()


@dataclass(slots=True)
class SparseLinSym(LinSym):
    """A class of two tensors representing sparse linear constraints."""

    ori_shape: tuple[int, ...]
    r_indices: Tensor
    c_indices: Tensor

    def detach(self) -> "SparseLinSym":
        return SparseLinSym(
            ori_shape=tuple(s for s in self.ori_shape),
            r_indices=self.r_indices.clone(),
            c_indices=self.c_indices.clone(),
            A=self.A.detach(),
            b=self.b.detach(),
        )

    def clone(self) -> "SparseLinSym":
        return SparseLinSym(
            ori_shape=tuple(s for s in self.ori_shape),
            r_indices=self.r_indices.clone(),
            c_indices=self.c_indices.clone(),
            A=self.A.clone(),
            b=self.b.clone(),
        )

    def to(self, *args, **kwargs) -> "SparseLinSym":
        device = kwargs.get("device", self.r_indices.device)
        return SparseLinSym(
            ori_shape=tuple(s for s in self.ori_shape),
            r_indices=self.r_indices.to(device=device),
            c_indices=self.c_indices.to(device=device),
            A=self.A.to(*args, **kwargs),
            b=self.b.to(*args, **kwargs),
        )

    def resparse(
        self, r_indices: Tensor | None = None, c_indices: Tensor | None = None
    ) -> "SparseLinSym":
        assert r_indices.numel() <= self.r_indices.numel()
        assert c_indices.numel() <= self.c_indices.numel()

        if (
            r_indices.numel() == self.r_indices.numel()
            and c_indices.numel() == self.c_indices.numel()
        ):
            return self

        dtype, device = self.A.dtype, self.A.device
        A, b = self.A, self.b
        if r_indices.numel() < self.r_indices.numel():
            new_r_mask = torch.zeros(self.ori_shape[0], dtype=torch.bool, device=device)
            new_r_mask[r_indices] = True
            new_r_mask = new_r_mask[self.r_indices]
            A = A[new_r_mask]
            b = b[new_r_mask]

        if c_indices.numel() < self.c_indices.numel():
            new_c_mask = torch.zeros(self.ori_shape[1], dtype=torch.bool, device=device)
            new_c_mask[c_indices] = True
            new_c_mask = new_c_mask[self.c_indices]
            A = A[:, new_c_mask]

        return SparseLinSym(
            ori_shape=self.ori_shape,
            r_indices=r_indices,
            c_indices=c_indices,
            A=A,
            b=b,
        )

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"shape={self.ori_shape}, "
            f"row_indices.shape={self.r_indices.shape}, "
            f"col_indices.shape={self.c_indices.shape}, "
            f"A.shape={self.A.shape}, "
            f"b.shape={self.b.shape}"
            f")"
        )

    def __repr__(self):
        return self.__str__()


@dataclass(slots=True)
class QuadSym(LinSym):
    """A class of two tensors representing quadratic constraints."""

    Q: Tensor

    def __post_init__(self):
        assert self.Q.shape[0] == self.b.shape[0]

    def to(self, *args, **kwargs) -> "QuadSym":
        """This is a similar function to the `torch.Tensor.to()` function."""
        return QuadSym(
            Q=self.Q.to(*args, **kwargs),
            A=self.A.to(*args, **kwargs),
            b=self.b.to(*args, **kwargs),
        )

    def clone(self) -> "QuadSym":
        """This is a similar function to the `torch.Tensor.clone()` function."""
        return QuadSym(
            Q=self.Q.clone(),
            A=self.A.clone(),
            b=self.b.clone(),
        )

    def detach(self) -> "QuadSym":
        """This is a similar function to the `torch.Tensor.detach()` function."""
        return QuadSym(
            Q=self.Q.detach(),
            A=self.A.detach(),
            b=self.b.detach(),
        )

    def __add__(self, other: "QuadSym") -> "QuadSym":
        return QuadSym(
            Q=self.Q + other.Q,
            A=self.A + other.A,
            b=self.b + other.b,
        )

    def __sub__(self, other: "QuadSym") -> "QuadSym":
        return QuadSym(
            Q=self.Q - other.Q,
            A=self.A - other.A,
            b=self.b - other.b,
        )

    def __iadd__(self, other: "QuadSym"):
        self.Q += other.Q
        self.A += other.A
        self.b += other.b
        return self

    def __isub__(self, other: "QuadSym"):
        self.Q -= other.Q
        self.A -= other.A
        self.b -= other.b
        return self

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"Q.shape={self.Q.shape}, "
            f"A.shape={self.A.shape}, "
            f"b={self.b.shape}"
            f")"
        )


@dataclass(slots=True)
class SymBound:
    """A class of two constraints representing lower and upper bounds."""

    L: LinSym | SparseLinSym | QuadSym
    U: LinSym | SparseLinSym | QuadSym | None

    def to(self, *args, **kwargs) -> "SymBound":
        """This is a similar function to the `torch.Tensor.to()` function."""
        return SymBound(
            L=self.L.to(*args, **kwargs),
            U=self.U.to(*args, **kwargs) if self.U is not None else None,
        )

    def clone(self) -> "SymBound":
        """This is a similar function to the `torch.Tensor.clone()` function."""
        return SymBound(
            L=self.L.clone(),
            U=self.U.clone() if self.U is not None else None,
        )

    def detach(self) -> "SymBound":
        """This is a similar function to the `torch.Tensor.detach()` function."""
        return SymBound(
            L=self.L.detach(), U=self.U.detach() if self.U is not None else None
        )

    def resparse(
        self, r_indices: Tensor | None = None, c_indices: Tensor | None = None
    ) -> "SymBound":
        return SymBound(
            L=self.L.resparse(r_indices, c_indices),
            U=self.U.resparse(r_indices, c_indices) if self.U is not None else None,
        )

    def __add__(self, other: "SymBound") -> "SymBound":
        return SymBound(
            L=self.L + other.L,
            U=None if self.U is None or other.U is None else self.U + other.U,
        )

    def __sub__(self, other: "SymBound") -> "SymBound":
        return SymBound(
            L=self.L - other.L,
            U=None if self.U is None or other.U is None else self.U - other.U,
        )

    def __iadd__(self, other: "SymBound"):
        self.L += other.L
        if self.U is not None and other.U is not None:
            self.U += other.U
        else:
            self.U = None
        return self

    def __isub__(self, other: "SymBound"):
        self.L -= other.L
        if self.U is not None and other.U is not None:
            self.U -= other.U
        else:
            self.U = None
        return self

    def __str__(self):
        return f"{self.__class__.__name__}(L={self.L}, U={self.U})"

    def __repr__(self):
        return self.__str__()
