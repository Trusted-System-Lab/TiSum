__all__ = [
    "ReLULinPropNode",
    "SigmoidLinPropNode",
    "TanhLinPropNode",
]

from src.nodes.basic import (
    ReLUNode,
    SigmoidNode,
    TanhNode,
)
from src.nodes.linprop._basic import NonlinearLinPropNode, UnaryOpNode
from src.utils import store_and_update_bnd
from src.substorch import (
    relax_relu,
    relax_sigmoid,
    relax_tanh,
    interval_relu,
    interval_sigmoid,
    interval_tanh,
    backsub_relu,
    backsub_sigmoid,
    backsub_tanh,
    backsub_relu_lower,
    backsub_sigmoid_lower,
    backsub_tanh_lower,
)


class ReLULinPropNode(ReLUNode, NonlinearLinPropNode, UnaryOpNode):

    def _build_rlx(self):
        assert len(self.pre_nodes) == 1

        pre_bound = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        relaxation = relax_relu(*pre_bound)
        self.cache.rlxs[self.name] = relaxation

    def _fwdprop_interval(self):
        assert len(self.pre_nodes) == 1

        pre_bound = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        new_bound = interval_relu(*pre_bound)
        store_and_update_bnd(new_bound, self.cache, self)

    def _bwdprop_symbnd(self):
        symbnd = self._get_bwdprop_symbnd()
        relaxation = self.cache.rlxs[self.name]
        backsub = backsub_relu if len(symbnd) == 4 else backsub_relu_lower
        new_symbnd = backsub(*symbnd, *relaxation)
        self.cache.symbnds[self.name] = new_symbnd


class SigmoidLinPropNode(SigmoidNode, NonlinearLinPropNode, UnaryOpNode):

    def _build_rlx(self):
        assert len(self.pre_nodes) == 1

        pre_bound = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        relaxation = relax_sigmoid(*pre_bound, mode=self.argument.act_relax_mode)
        self.cache.rlxs[self.name] = relaxation

    def _fwdprop_interval(self):
        assert len(self.pre_nodes) == 1

        pre_bound = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        new_bound = interval_sigmoid(*pre_bound)
        store_and_update_bnd(new_bound, self.cache, self)

    def _bwdprop_symbnd(self):
        symbnd = self._get_bwdprop_symbnd()
        relaxation = self.cache.rlxs[self.name]
        backsub = backsub_sigmoid if len(symbnd) == 4 else backsub_sigmoid_lower
        new_symbnd = backsub(*symbnd, *relaxation)
        self.cache.symbnds[self.name] = new_symbnd


class TanhLinPropNode(TanhNode, NonlinearLinPropNode, UnaryOpNode):

    def _build_rlx(self):
        assert len(self.pre_nodes) == 1

        pre_bound = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        relaxation = relax_tanh(*pre_bound, mode=self.argument.act_relax_mode)
        self.cache.rlxs[self.name] = relaxation

    def _fwdprop_interval(self):
        assert len(self.pre_nodes) == 1

        pre_bound = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        new_bound = interval_tanh(*pre_bound)
        store_and_update_bnd(new_bound, self.cache, self)

    def _bwdprop_symbnd(self):
        symbnd = self._get_bwdprop_symbnd()
        relaxation = self.cache.rlxs[self.name]
        backsub = backsub_tanh if len(symbnd) == 4 else backsub_tanh_lower
        new_symbnd = backsub(*symbnd, *relaxation)
        self.cache.symbnds[self.name] = new_symbnd
