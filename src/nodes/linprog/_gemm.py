__all__ = ["GEMMLinProgNode"]

from src.miplayer import add_constrs_gemm
from src.args import LinProgArgs
from src.cache import LinProgCache
from src.nodes.basic import GEMMNode
from src.nodes.linprog._basic import BasicLinProgNode


class GEMMLinProgNode(GEMMNode, BasicLinProgNode):
    argument: LinProgArgs
    cache: LinProgCache

    def forward(self):
        BasicLinProgNode.forward(self)

    def add_constrs(self):
        model = self.cache.lp_model
        x = self.cache.lp_vars[self.pre_nodes[0].name]
        y = self.cache.lp_vars[self.name]
        weight = self.weight.cpu().numpy()
        bias = self.bias.cpu().numpy()
        constrs_list = add_constrs_gemm(model, x, y, weight, bias, name=self.name)

        self.cache.lp_constrs[self.name] = constrs_list
