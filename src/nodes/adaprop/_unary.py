__all__ = [
    "ReLUAdaPropNode",
    "SigmoidAdaPropNode",
    "TanhAdaPropNode",
]

from src.cache import AdaPropCache
from src.nodes.linprop import ReLULinPropNode, SigmoidLinPropNode, TanhLinPropNode
from src.substorch import (
    ada_relax_relu,
    ada_relax_sigmoid,
    ada_relax_tanh,
    get_rlx_mask_relu,
    get_rlx_mask_sigmoid,
    get_rlx_mask_tanh,
    ada_backsub_relu,
    ada_backsub_relu_lower,
    ada_backsub_sigmoid,
    ada_backsub_sigmoid_lower,
    ada_backsub_tanh,
    ada_backsub_tanh_lower,
)


class ReLUAdaPropNode(ReLULinPropNode):
    cache: AdaPropCache

    def _build_rlx(self):
        assert len(self.pre_nodes) == 1

        pre_bound = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        pre_lower_bound, pre_upper_bound = pre_bound
        self._fwdprop_interval()
        lower_bound, upper_bound = self.cache.bnds[self.name]
        ada_mask = get_rlx_mask_relu(
            pre_lower_bound,
            pre_upper_bound,
            lower_bound,
            upper_bound,
            min_range=self.argument.act_relax_min_range,
        )
        relaxation, lower_ada_idx, lower_ada_param = ada_relax_relu(
            *pre_bound, ada_mask=ada_mask
        )
        self.cache.rlxs[self.name] = relaxation
        self.cache.ada_rlx_idxs[self.name] = (lower_ada_idx, None)
        self.cache.ada_rlx_prms[self.name] = (lower_ada_param, None)

    def _bwdprop_symbnd(self):
        symbnd = self._get_bwdprop_symbnd()
        relaxation = self.cache.rlxs[self.name]
        backsub = ada_backsub_relu if len(symbnd) == 4 else ada_backsub_relu_lower
        lower_ada_idx = self.cache.ada_rlx_idxs[self.name][0]
        lower_ada_param = self.cache.ada_rlx_prms[self.name][0]

        new_symbnd = backsub(
            *symbnd,
            *relaxation,
            lower_ada_idx=lower_ada_idx,
            lower_ada_param=lower_ada_param
        )
        self.cache.symbnds[self.name] = new_symbnd


class SigmoidAdaPropNode(SigmoidLinPropNode):
    cache: AdaPropCache

    def _build_rlx(self):
        assert len(self.pre_nodes) == 1

        pre_bound = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        pre_lower_bound, pre_upper_bound = pre_bound
        self._fwdprop_interval()
        lower_bound, upper_bound = self.cache.bnds[self.name]
        ada_mask = get_rlx_mask_sigmoid(
            pre_lower_bound,
            pre_upper_bound,
            lower_bound,
            upper_bound,
            min_range=self.argument.act_relax_min_range,
            sigmoid_relax_limit=self.argument.sigmoid_relax_limit,
        )
        relaxation, ada_idx, ada_range, ada_param = ada_relax_sigmoid(
            *pre_bound, ada_mask=ada_mask
        )
        self.cache.rlxs[self.name] = relaxation
        self.cache.ada_rlx_idxs[self.name] = ada_idx
        self.cache.ada_rlx_ranges[self.name] = ada_range
        self.cache.ada_rlx_prms[self.name] = ada_param

    def _bwdprop_symbnd(self):
        symbnd = self._get_bwdprop_symbnd()
        relaxation = self.cache.rlxs[self.name]
        backsub = ada_backsub_sigmoid if len(symbnd) == 4 else ada_backsub_sigmoid_lower
        ada_idx = self.cache.ada_rlx_idxs[self.name]
        ada_range = self.cache.ada_rlx_ranges[self.name]
        ada_param = self.cache.ada_rlx_prms[self.name]
        new_symbnd = backsub(
            *symbnd,
            *relaxation,
            ada_idx=ada_idx,
            ada_range=ada_range,
            ada_param=ada_param
        )
        self.cache.symbnds[self.name] = new_symbnd


class TanhAdaPropNode(TanhLinPropNode):
    cache: AdaPropCache

    def _build_rlx(self):
        assert len(self.pre_nodes) == 1

        pre_bound = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        pre_lower_bound, pre_upper_bound = pre_bound
        self._fwdprop_interval()
        lower_bound, upper_bound = self.cache.bnds[self.name]
        ada_mask = get_rlx_mask_tanh(
            pre_lower_bound,
            pre_upper_bound,
            lower_bound,
            upper_bound,
            min_range=self.argument.act_relax_min_range,
            tanh_relax_limit=self.argument.tanh_relax_limit,
        )
        relaxation, ada_idx, ada_range, ada_param = ada_relax_tanh(
            *pre_bound, ada_mask=ada_mask
        )
        self.cache.rlxs[self.name] = relaxation

    def _bwdprop_symbnd(self):
        symbnd = self._get_bwdprop_symbnd()
        relaxation = self.cache.rlxs[self.name]
        backsub = ada_backsub_tanh if len(symbnd) == 4 else ada_backsub_tanh_lower
        ada_idx = self.cache.ada_rlx_idxs[self.name]
        ada_range = self.cache.ada_rlx_ranges[self.name]
        ada_param = self.cache.ada_rlx_prms[self.name]
        new_symbnd = backsub(
            *symbnd,
            *relaxation,
            ada_idx=ada_idx,
            ada_range=ada_range,
            ada_param=ada_param
        )
        self.cache.symbnds[self.name] = new_symbnd
