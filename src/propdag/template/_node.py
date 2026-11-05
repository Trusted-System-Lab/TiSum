__all__ = ["TNode"]

from abc import ABC

from ._arguments import TArgument
from ._cache import TCache


class TNode(ABC):
    """Abstract base class for nodes in a computational graph."""

    _name: str
    _cache: TCache
    _argument: TArgument
    _pre_nodes: list["TNode"]
    _next_nodes: list["TNode"]

    def __init__(self, name: str, cache: TCache, argument: TArgument):
        """Initialize a node in the computational graph."""
        self._name = name
        self._cache = cache
        self._argument = argument
        self._pre_nodes = []
        self._next_nodes = []

    def forward(self):
        """Perform forward computation for this node."""
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

    def backward(self):
        """Perform backward computation for this node."""
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

    def clear_fwd_cache(self):
        """Clear forward computation cache."""
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

    def clear_bwd_cache(self):
        """Clear backward computation cache."""
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

    def _init_symbnd(self):
        """Initialize symbolic bounds."""
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

    def _build_rlx(self):
        """Build relaxation for this node's operation."""
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

    def _fwdprop_symbnd(self):
        """Forward propagate symbolic bounds."""
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

    def _bwdprop_symbnd(self):
        """Backward propagate symbolic bounds."""
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

    def _cal_and_update_cur_node_bnd(self):
        """Calculate and update bounds for current node."""
        raise RuntimeError(
            f"This method should be instantiated in {type(self).__name__}."
        )

    @property
    def name(self):
        """Get node name."""
        return self._name

    @property
    def cache(self) -> TCache:
        """Get shared cache instance."""
        return self._cache

    @cache.setter
    def cache(self, value: TCache):
        """Set shared cache instance."""
        self._cache = value

    @property
    def argument(self) -> TArgument:
        """Get shared arguments instance."""
        return self._argument

    @argument.setter
    def argument(self, value: TArgument):
        """Set shared arguments instance."""
        self._argument = value

    @property
    def pre_nodes(self) -> list["TNode"]:
        """Get predecessor nodes."""
        return self._pre_nodes

    @pre_nodes.setter
    def pre_nodes(self, value: list["TNode"]):
        """Set predecessor nodes."""
        self._pre_nodes = value

    @property
    def next_nodes(self) -> list["TNode"]:
        """Get successor nodes."""
        return self._next_nodes

    @next_nodes.setter
    def next_nodes(self, value: list["TNode"]):
        """Set successor nodes."""
        self._next_nodes = value
