__all__ = ["BatchedSliceLinPropNode"]

from src.nodes.basic import SliceNode
from src.nodes.batched_linprop._basic import BatchedBasicLinPropNode
from src.utils import batched_store_and_update_bnd
from src.substorch import (
    batched_linear_backsub_slice,
    batched_linear_backsub_slice_lower,
    batched_interval_slice,
)


class BatchedSliceLinPropNode(SliceNode, BatchedBasicLinPropNode):

    def forward(self):
        BatchedBasicLinPropNode.forward(self)
        assert len(self.pre_nodes) == 1

        pre_bound = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        new_bound = batched_interval_slice(
            *pre_bound,
            starts=self.starts,
            ends=self.ends,
            axes=self.axes,
            steps=self.steps,
        )

        batched_store_and_update_bnd(new_bound, self.cache, self)

    def _bwdprop_symbnd(self):
        symbnd = self._get_bwdprop_symbnd()
        backsub = (
            batched_linear_backsub_slice
            if len(symbnd) == 4
            else (batched_linear_backsub_slice_lower)
        )
        new_symbnd = backsub(
            *symbnd,
            ori_shape=self.input_sizes[0],
            starts=self.starts,
            ends=self.ends,
            axes=self.axes,
            steps=self.steps,
        )

        self.cache.symbnds[self.name] = new_symbnd
