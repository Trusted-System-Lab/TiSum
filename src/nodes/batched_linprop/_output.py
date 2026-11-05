__all__ = ["BatchedOutputLinPropNode"]

from src.cache import LinPropCache
from src.containers import OutputNodeInfo
from src.nodes.basic import OutputNode
from src.nodes.batched_linprop._gemm import BatchedGEMMLinPropNode
from src.substorch import batched_init_symbnd_gemm


class BatchedOutputLinPropNode(OutputNode, BatchedGEMMLinPropNode):
    cache: LinPropCache

    def _unpack_node_info(self, node_info: OutputNodeInfo):
        OutputNode._unpack_node_info(self, node_info)

    def _init_symbnd(self):
        """This method is similar to linear nodes but only considering the lower bound."""
        symbnd = batched_init_symbnd_gemm(
            self.weight, self.bias, n_batches=self.batch_size
        )
        symbnd_coeffs, symbnd_const = symbnd

        new_symbnd = (
            symbnd_coeffs,
            symbnd_const,
            symbnd_coeffs.clone().detach(),
            symbnd_const.clone().detach(),
        )
        self.cache.symbnds[self.name] = new_symbnd
