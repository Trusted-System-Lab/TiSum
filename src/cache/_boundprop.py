__all__ = [
    "BasicCache",
    "LinPropCache",
    "QuadPropCache",
    "AdaPropCache",
    "BaBPropCache",
]

from dataclasses import dataclass, field

import torch
from torch import Tensor

from propdag import TCache
from src.containers import *


@dataclass(slots=True)
class BasicCache(TCache):
    dtype: torch.dtype
    device: torch.device
    random_seed: int
    bnds: dict[str, tuple[Tensor, ...] | tuple[tuple[Tensor, ...], ...]] = field(
        default_factory=dict, repr=False
    )

    def clear(self, keep_bnds: bool = False):
        if not keep_bnds:
            self.bnds = {}

    def inherit(self, cache: "BasicCache"):
        self.bnds = cache.bnds


@dataclass(slots=True)
class LinPropCache(BasicCache):
    cur_node: "BasicNode" = None
    rlxs: dict[str, tuple[Tensor, ...]] = field(default_factory=dict, repr=False)
    symbnds: dict[str, tuple[Tensor, ...] | tuple[tuple[Tensor, ...], ...]] = field(
        default_factory=dict, repr=False
    )
    output_final_symbnds: tuple[Tensor, ...] | None = None

    def clear(
        self,
        keep_bnds: bool = False,
        keep_rlxs: bool = False,
        keep_symbnds: bool = False,
        keep_output_final_symbnds: bool = False,
    ):
        BasicCache.clear(self, keep_bnds)
        if not keep_rlxs:
            self.rlxs = {}
        if not keep_symbnds:
            self.symbnds = {}
        if not keep_output_final_symbnds:
            self.output_final_symbnds = None

    def inherit(self, cache: "LinPropCache"):
        BasicCache.inherit(self, cache)
        self.rlxs = cache.rlxs
        self.output_final_symbnds = cache.output_final_symbnds


@dataclass(slots=True)
class QuadPropCache(LinPropCache):
    pass


@dataclass(slots=True)
class AdaPropCache(LinPropCache):
    ada_rlx_idxs: dict[str, tuple[Tensor | None, Tensor | None]] = field(
        default_factory=dict, repr=False
    )
    ada_rlx_prms: dict[str, tuple[Tensor | None, Tensor | None]] = field(
        default_factory=dict, repr=False
    )
    ada_rlx_ranges: dict[
        str, tuple[tuple[Tensor, Tensor] | None, tuple[Tensor, Tensor] | None]
    ] = field(default_factory=dict, repr=False)

    def clear(
        self,
        keep_bnds: bool = False,
        keep_rlxs: bool = False,
        keep_symbnds: bool = False,
        keep_output_final_symbnds: bool = False,
    ):
        LinPropCache.clear(
            self, keep_bnds, keep_rlxs, keep_symbnds, keep_output_final_symbnds
        )
        self.ada_rlx_idxs = {}
        self.ada_rlx_prms = {}
        self.ada_rlx_ranges = {}

    def detach(self):
        for k, v in self.bnds.items():
            self.bnds[k] = tuple(x.detach() for x in v)
        for k, v in self.rlxs.items():
            self.rlxs[k] = tuple(x.detach() for x in v)


@dataclass(slots=True)
class BaBPropCache(AdaPropCache):
    substitutions: dict[str, SymBound] = field(default_factory=dict, repr=False)

    def clear(
        self,
        keep_bnds: bool = False,
        keep_rlxs: bool = False,
        keep_symbnds: bool = False,
        keep_output_final_symbnds: bool = False,
        keep_substitutions: bool = False,
    ):
        AdaPropCache.clear(
            self, keep_bnds, keep_rlxs, keep_symbnds, keep_output_final_symbnds
        )
        if not keep_substitutions:
            self.substitutions = {}
