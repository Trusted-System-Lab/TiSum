__all__ = ["FlattenLinProgNode"]

from src.args import LinProgArgs
from src.cache import LinProgCache
from src.nodes.basic import FlattenNode
from src.nodes.linprog._basic import BasicLinProgNode


class FlattenLinProgNode(FlattenNode, BasicLinProgNode):
    argument: LinProgArgs
    cache: LinProgCache

    def forward(self):
        BasicLinProgNode.forward(self)

    def add_vars(self):
        self.cache.lp_vars[self.name] = self.cache.lp_vars[self.pre_nodes[0].name]

    def add_constrs(self):
        pass
