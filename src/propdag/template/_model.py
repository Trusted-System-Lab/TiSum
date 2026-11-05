__all__ = [
    "TModel",
    "clear_fwd_cache",
    "clear_bwd_cache",
]


from abc import ABC
from typing import Literal

from ._arguments import TArgument
from ._cache import TCache
from ._node import TNode
from ._sort import *
from ..utils import *


def clear_fwd_cache(cache_counter: dict[TNode, int], nodes: list[TNode]):
    """Clear forward caches for nodes when they are no longer needed."""
    for node in nodes:
        cache_counter[node] -= 1
        if cache_counter[node] <= 0:
            node.clear_fwd_cache()
            del cache_counter[node]


def clear_bwd_cache(cache_counter: dict[TNode, int], nodes: list[TNode]):
    """Clear backward caches for nodes when they are no longer needed."""
    for node in nodes:
        if node in cache_counter:
            cache_counter[node] -= 1
            if cache_counter[node] <= 0:
                node.clear_bwd_cache()
                del cache_counter[node]


class TModel(ABC):
    """Template for computational graph model."""

    _nodes: list[TNode]
    _sort_strategy: Literal["dfs", "bfs"]
    _cache: TCache
    _arguments: TArgument
    _all_backward_sorts: dict[TNode, list[TNode]]

    verbose: bool

    def __init__(
        self,
        nodes: list[TNode],
        sort_strategy: Literal["dfs", "bfs"] = "bfs",
        verbose: bool = False,
    ):
        """Initialize a computational graph model."""
        self.verbose = verbose
        self._sort_strategy = sort_strategy
        if sort_strategy == "dfs":
            self._nodes = topo_sort_forward_dfs(nodes, self.verbose)
        elif sort_strategy == "bfs":
            self._nodes = topo_sort_forward_bfs(nodes, self.verbose)
        else:
            raise ValueError(f"Unknown sort strategy: {sort_strategy}")
        self._cache = nodes[0].cache
        self._arguments = nodes[0].argument
        for node in nodes[1:]:
            assert node.cache == self._cache
            assert node.argument == self._arguments

        if self.arguments.prop_mode == PropMode.BACKWARD:
            self._all_backward_sorts = topo_sort_backward(self.nodes, self.verbose)

    def run(self, *args, **kwargs):
        """Execute the computational graph."""
        cache_counter = {node: len(node.next_nodes) for node in self._nodes}

        node = self.nodes[0]
        if self.verbose:
            print(f"Forward pass {node.name}")
        node.forward()

        for i in range(1, len(self.nodes)):
            node = self.nodes[i]
            if self.verbose:
                print(f"Forward pass {node.name}")
            node.forward()

            if self.arguments.prop_mode == PropMode.BACKWARD:
                self.backsub(node)

            clear_fwd_cache(cache_counter, node.pre_nodes)

        clear_fwd_cache(cache_counter, [node])

    def backsub(self, node: TNode):
        """Perform back-substitution from a specified node."""
        backward_sort = self._all_backward_sorts[node]

        if len(backward_sort) == 1:
            return

        cache_counter = {node: len(node.pre_nodes) for node in backward_sort}

        if self.verbose:
            print(f"\tBack-substitute {node.name}")
        node.backward()

        for j in range(1, len(backward_sort)):
            node = backward_sort[j]
            if self.verbose:
                print(f"\tBack-substitute {node.name}")
            node.backward()
            clear_bwd_cache(cache_counter, node.next_nodes)

        clear_bwd_cache(cache_counter, [node])

    @property
    def sort_strategy(self):
        """Get the sorting strategy used in the model."""
        return self._sort_strategy

    @property
    def nodes(self):
        """Get the nodes in the model."""
        return self._nodes

    @property
    def cache(self) -> TCache:
        """Get the shared cache for the model."""
        return self._cache

    @property
    def arguments(self) -> TArgument:
        """Get the shared arguments for the model."""
        return self._arguments
