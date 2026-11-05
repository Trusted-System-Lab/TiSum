__all__ = ["BatchedFlattenLinPropNode"]

from src.nodes.basic import (
    FlattenNode,
)
from src.nodes.batched_linprop._basic import BatchedShapingLinPropNode
from src.utils import batched_store_and_update_bnd
from src.substorch import (
    batched_linear_backsub_flatten,
    batched_linear_backsub_flatten_lower,
    batched_interval_flatten,
)


class BatchedFlattenLinPropNode(FlattenNode, BatchedShapingLinPropNode):

    def forward(self):
        assert len(self.pre_nodes) == 1
        pre_bound = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        new_bound = batched_interval_flatten(*pre_bound, ori_shape=self.output_sizes[0])
        batched_store_and_update_bnd(new_bound, self.cache, self)

    def _bwdprop_symbnd(self):
        assert len(self.next_nodes) == 1

        symbnd = self._get_bwdprop_symbnd()
        backsub = (
            batched_linear_backsub_flatten
            if len(symbnd) == 4
            else batched_linear_backsub_flatten_lower
        )
        new_symbnd = backsub(*symbnd, ori_shape=self.input_sizes[0])
        self.cache.symbnds[self.name] = new_symbnd
