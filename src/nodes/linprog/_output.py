__all__ = ["OutputLinProgNode"]

import mip

from src.cache import LinProgCache
from src.containers import OutputNodeInfo
from src.nodes.basic import OutputNode
from src.nodes.linprog._gemm import GEMMLinProgNode


class OutputLinProgNode(OutputNode, GEMMLinProgNode):
    cache: LinProgCache

    def forward(self):
        GEMMLinProgNode.forward(self)

    def _unpack_node_info(self, node_info: OutputNodeInfo):
        OutputNode._unpack_node_info(self, node_info)

    def add_vars(self):
        model = self.cache.lp_model
        n = self.weight.shape[1]
        vars_list = []

        for i in range(n):
            var = model.add_var(name=f"{self.name}_{i}", lb=-mip.INF, ub=mip.INF)
            vars_list.append(var)

        self.cache.lp_vars[self.name] = vars_list
