__all__ = ["ToyModel"]

from ._arguments import ToyArgument
from ._cache import ToyCache
from ..template import *


class ToyModel(TModel):
    """A toy implementation of the template model."""

    _nodes: list[TNode]
    _cache: ToyCache
    _arguments: ToyArgument
    _all_backward_sorts: dict[TNode, list[TNode]]

    def run(self):
        """Execute the toy model with visible logging."""
        print("Running ToyModel...")
        super().run()
