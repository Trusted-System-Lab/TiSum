__all__ = ["BatchedInputLinPropNode"]

from src.nodes.basic import InputNode
from src.nodes.batched_linprop._basic import BatchedShapingLinPropNode
from src.substorch import (
    batched_linear_backsub_flatten,
    batched_linear_backsub_flatten_lower,
)


class BatchedInputLinPropNode(InputNode, BatchedShapingLinPropNode):

    def forward(self):
        assert self.name in self.cache.bnds

    def backward(self, store_bnd: bool = True):
        BatchedShapingLinPropNode.backward(self, store_bnd)
        if store_bnd:
            self._cal_and_update_cur_node_bnd()

    def _bwdprop_symbnd(self):
        symbnd = self._get_bwdprop_symbnd()
        backsub = (
            batched_linear_backsub_flatten
            if len(symbnd) == 4
            else (batched_linear_backsub_flatten_lower)
        )
        new_symbnd = backsub(*symbnd, ori_shape=self.output_sizes[0])
        self.cache.symbnds[self.name] = new_symbnd
