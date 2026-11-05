__all__ = [
    "AddLinPropNode",
    "AddCstLinPropNode",
]

from src.nodes.basic import (
    AddNode,
    AddCstNode,
)
from src.nodes.linprop._basic import (
    UnaryOpNode,
    BinaryOpNode,
    LinearLinPropNode,
)
from src.utils import store_and_update_bnd
from src.substorch import (
    interval_add,
    interval_addcst,
    backsub_add,
    backsub_add_lower,
    backsub_addcst,
    backsub_addcst_lower,
)


class AddLinPropNode(AddNode, LinearLinPropNode, BinaryOpNode):

    def _fwdprop_interval(self):
        assert len(self.pre_nodes) == 2

        pre_bound1 = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        pre_bound2 = self._get_fwdprop_bnd(self.pre_nodes[1], self.input_names[1])
        new_bound = interval_add(*pre_bound1, *pre_bound2)
        store_and_update_bnd(new_bound, self.cache, self)

    def _bwdprop_symbnd(self):
        symbnd = self._get_bwdprop_symbnd()
        backsub = backsub_add if len(symbnd) == 4 else backsub_add_lower
        new_symbnd = backsub(*symbnd)
        new_symbnd = self._reduce_broadcast_after_backsub(new_symbnd)
        self.cache.symbnds[self.name] = new_symbnd


class AddCstLinPropNode(AddCstNode, LinearLinPropNode, UnaryOpNode):

    def _init_symbnd(self):
        raise RuntimeError(
            "This node does not need to initialize symbolic bounds "
            "but use forward interval arithmetic to calculate its bounds."
        )

    def _fwdprop_interval(self):
        assert len(self.pre_nodes) == 1

        pre_bound = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        new_bound = interval_addcst(*pre_bound, constant=self.constant)
        store_and_update_bnd(new_bound, self.cache, self)

    def _bwdprop_symbnd(self):
        symbnd = self._get_bwdprop_symbnd()
        backsub = backsub_addcst if len(symbnd) == 4 else backsub_addcst_lower
        new_symbnd = backsub(*symbnd, constant=self.constant)
        new_symbnd = self._reduce_broadcast_after_backsub(new_symbnd)
        self.cache.symbnds[self.name] = new_symbnd
