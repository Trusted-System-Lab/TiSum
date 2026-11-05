__all__ = ["SliceLinPropNode"]

from src.nodes.basic import SliceNode
from src.nodes.linprop._basic import BasicLinPropNode
from src.utils import store_and_update_bnd
from src.substorch import (
    backsub_slice,
    backsub_slice_lower,
    interval_slice,
)


class SliceLinPropNode(SliceNode, BasicLinPropNode):

    def forward(self):
        BasicLinPropNode.forward(self)
        assert len(self.pre_nodes) == 1

        pre_bound = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        new_bound = interval_slice(
            *pre_bound,
            starts=self.starts,
            ends=self.ends,
            axes=self.axes,
            steps=self.steps,
        )
        store_and_update_bnd(new_bound, self.cache, self)

    def _bwdprop_symbnd(self):
        symbnd = self._get_bwdprop_symbnd()
        backsub = backsub_slice if len(symbnd) == 4 else backsub_slice_lower
        new_symbnd = backsub(
            *symbnd,
            ori_shape=self.input_sizes[0],
            starts=self.starts,
            ends=self.ends,
            axes=self.axes,
            steps=self.steps,
        )
        self.cache.symbnds[self.name] = new_symbnd
