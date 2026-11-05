__all__ = ["AddLinProgNode"]

from src.miplayer import add_constrs_add
from src.args import LinProgArgs
from src.cache import LinProgCache
from src.nodes.basic import AddNode
from src.nodes.linprog._basic import BasicLinProgNode


class AddLinProgNode(AddNode, BasicLinProgNode):
    argument: LinProgArgs
    cache: LinProgCache

    def forward(self):
        BasicLinProgNode.forward(self)

    def add_constrs(self):
        model = self.cache.lp_model
        x1 = self.cache.lp_vars[self.pre_nodes[0].name]
        x2 = self.cache.lp_vars[self.pre_nodes[1].name]
        y = self.cache.lp_vars[self.name]
        constrs_list = add_constrs_add(model, x1, x2, y, name=self.name)

        self.cache.lp_constrs[self.name] = constrs_list
