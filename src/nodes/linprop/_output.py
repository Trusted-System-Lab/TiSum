__all__ = ["OutputLinPropNode"]

from src.containers import OutputNodeInfo
from src.nodes.basic import OutputNode
from src.nodes.linprop._gemm import GEMMLinPropNode
from src.substorch import init_symbnd_gemm


class OutputLinPropNode(OutputNode, GEMMLinPropNode):

    def _unpack_node_info(self, node_info: OutputNodeInfo):
        OutputNode._unpack_node_info(self, node_info)

    def _init_symbnd(self):
        """This method is similar to linear nodes but only considering the lower bound."""
        symbnd = init_symbnd_gemm(self.weight, self.bias)
        symbnd_coeffs, symbnd_const = symbnd

        new_symbnd = (
            symbnd_coeffs,
            symbnd_const,
            symbnd_coeffs.clone().detach(),
            symbnd_const.clone().detach(),
        )
        self.cache.symbnds[self.name] = new_symbnd
