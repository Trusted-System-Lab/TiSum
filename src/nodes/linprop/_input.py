__all__ = ["InputLinPropNode"]

from src.nodes.basic import InputNode
from src.nodes.linprop._basic import ShapingLinPropNode
from src.substorch import backsub_reshape, backsub_reshape_lower


class InputLinPropNode(InputNode, ShapingLinPropNode):

    def forward(self):
        """This method only check if the input bounds in the cache."""

        assert self.name in self.cache.bnds

    def backward(self, store_bnd: bool = True):
        """The input node needs backward propagation and calculate the bounds."""

        ShapingLinPropNode.backward(self, store_bnd)
        if store_bnd:
            self._cal_and_update_cur_node_bnd()

    def _bwdprop_symbnd(self):
        symbnd = self._get_bwdprop_symbnd()
        backsub = backsub_reshape if len(symbnd) == 4 else backsub_reshape_lower
        new_symbnd = backsub(*symbnd, ori_shape=self.output_sizes[0])
        self.cache.symbnds[self.name] = new_symbnd
