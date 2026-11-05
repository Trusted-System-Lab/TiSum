__all__ = ["BackwardToyNode"]

from ._arguments import *
from ._cache import *
from ..template import *


class BackwardToyNode(TNode):
    """Node implementation for backward propagation in toy models."""

    _name: str
    _cache: ToyCache
    _argument: ToyArgument
    _pre_nodes: list["BackwardToyNode"]
    _next_nodes: list["BackwardToyNode"]

    def forward(self):
        """Perform forward initialization for backward propagation."""
        if len(self._pre_nodes) == 0:
            assert self.name in self.cache.bnds
            print(f"{self.name}: Skip input node")
            return

        self.cache.cur_node = self

        self._build_rlx()
        self._init_symbnd()

    def backward(self):
        """Perform backward propagation for this node."""
        self._bwdprop_symbnd()
        self._cal_and_update_cur_node_bnd()

    def clear_fwd_cache(self):
        """Clear forward computation caches for this node."""
        if len(self.next_nodes) > 0 and len(self.pre_nodes) > 0:
            print(f"{self.name}: Clear forward cache of bounds")
            del self.cache.bnds[self.name]

    def clear_bwd_cache(self):
        """Clear backward computation caches for this node."""
        print(f"{self.name}: Clear backforward cache of symbolic bounds")
        del self.cache.symbnds[self.name]

    @property
    def cache(self) -> ToyCache:
        """Get the toy cache instance."""
        return self._cache

    @property
    def argument(self) -> ToyArgument:
        """Get the toy argument instance."""
        return self._argument

    def _init_symbnd(self):
        """Initialize symbolic bounds for linear nodes."""
        print(f"{self.name}: Build symbolic bounds if this is a linear node")

    def _build_rlx(self):
        """Build relaxations for non-linear operations."""
        print(f"{self.name}: Calculate relaxation if this is a non-linear node")

    def _fwdprop_symbnd(self):
        """Not supported in backward propagation mode."""
        raise RuntimeError("Forward pass is not supported in backward mode. ")

    def _bwdprop_symbnd(self):
        """Backward propagate symbolic bounds."""
        if self == self.cache.cur_node:
            print(f"{self.name}: Prepare symbolic bounds of {self.name}")
        else:
            print(
                f"{self.name}: Backsubstitute symbolic bounds of {self.cache.cur_node.name}"
            )
        print(f"{self.name}: Cache substitution")
        self.cache.symbnds[self.name] = (f"substitution of {self.name}",)

    def _cal_and_update_cur_node_bnd(self):
        """Calculate and cache scalar bounds for the current node."""
        cur_name = self.cache.cur_node.name
        print(f"{self.name}: Calculate scalar bounds of {cur_name}")
        print(f"{self.name}: Cache scalar bounds")
        self.cache.bnds[cur_name] = ("scalar bounds",)
