__all__ = [
    "FlattenLinPropNode",
    "ReshapeLinPropNode",
]

from src.nodes.basic import (
    FlattenNode,
    ReshapeNode,
)
from src.nodes.linprop._basic import ShapingLinPropNode
from src.utils import store_and_update_bnd
from src.substorch import (
    backsub_flatten,
    backsub_reshape,
    backsub_flatten_lower,
    backsub_reshape_lower,
    interval_flatten,
    interval_reshape,
)


class FlattenLinPropNode(FlattenNode, ShapingLinPropNode):

    def forward(self):
        assert len(self.pre_nodes) == 1

        pre_bound = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        new_bound = interval_flatten(*pre_bound, ori_shape=self.output_sizes[0])
        store_and_update_bnd(new_bound, self.cache, self)

    def _bwdprop_symbnd(self):
        assert len(self.next_nodes) == 1

        symbnd = self._get_bwdprop_symbnd()
        backsub = backsub_flatten if len(symbnd) == 4 else backsub_flatten_lower
        new_symbnd = backsub(*symbnd, ori_shape=self.input_sizes[0])
        self.cache.symbnds[self.name] = new_symbnd


class ReshapeLinPropNode(ReshapeNode, ShapingLinPropNode):

    def forward(self):
        assert len(self.pre_nodes) == 1

        pre_bound = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        new_bound = interval_reshape(*pre_bound, ori_shape=self.output_sizes[0])
        store_and_update_bnd(new_bound, self.cache, self)

    def _bwdprop_symbnd(self):
        assert len(self.next_nodes) == 1

        symbnd = self._get_bwdprop_symbnd()
        backsub = backsub_reshape if len(symbnd) == 4 else backsub_reshape_lower
        new_symbnd = backsub(*symbnd, ori_shape=self.input_sizes[0])
        self.cache.symbnds[self.name] = new_symbnd
