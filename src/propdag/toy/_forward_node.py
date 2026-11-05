__all__ = ["ForwardToyNode"]

from ._arguments import *
from ._cache import *
from ..template import *


class ForwardToyNode(TNode):
    """Node implementation for forward propagation in toy models."""

    _name: str
    _cache: ToyCache
    _argument: ToyArgument
    _pre_nodes: list["ForwardToyNode"]
    _next_nodes: list["ForwardToyNode"]

    def forward(self):
        """Perform forward computation for this node."""
        if len(self._pre_nodes) == 0:
            assert self.name in self.cache.bnds
            print(f"{self.name}: Skip input node")
            return

        self.cache.cur_node = self

        self._build_rlx()
        self._fwdprop_symbnd()
        self._cal_and_update_cur_node_bnd()

    def backward(self):
        """Not supported in forward propagation mode."""
        raise RuntimeError("Backward pass is not supported in forward mode")

    def clear_fwd_cache(self):
        """Clear forward computation caches for this node."""
        if len(self.next_nodes) > 0 and len(self.pre_nodes) > 0:
            print(f"{self.name}: Clear forward cache of bounds")
            del self.cache.bnds[self.name]
        print(f"{self.name}: Clear forward cache of symbolic bounds")
        del self.cache.symbnds[self.name]

    def clear_bwd_cache(self):
        """Not supported in forward propagation mode."""
        raise RuntimeError("Backward pass is not supported in forward mode")

    @property
    def cache(self) -> ToyCache:
        """Get the toy cache instance."""
        return self._cache

    @property
    def argument(self) -> ToyArgument:
        """Get the toy argument instance."""
        return self._argument

    def _build_rlx(self):
        """Build relaxations for non-linear operations."""
        print(f"{self.name}: Calculate relaxation if this is non-linear node")

    def _fwdprop_symbnd(self):
        """Forward propagate symbolic bounds from predecessor nodes."""
        if len(self.pre_nodes) == 0:
            print(f"{self.name}: Prepare symbolic bounds of {self.name}")
        else:
            pre_names = [pre_node.name for pre_node in self.pre_nodes]
            print(f"{self.name}: Forward propagate symbolic bounds of {pre_names}")
        print(f"{self.name}: Cache symbolic bounds")
        self.cache.symbnds[self.name] = ("symbolic bounds",)

    def _bwdprop_symbnd(self):
        """Not supported in forward propagation mode."""
        raise RuntimeError("Backward pass is not supported in forward mode")

    def _cal_and_update_cur_node_bnd(self):
        """Calculate and cache scalar bounds for this node."""
        print(f"{self.name}: Calculate scalar bounds")
        print(f"{self.name}: Cache scalar bounds")
        self.cache.bnds[self.name] = ("scalar bounds",)
