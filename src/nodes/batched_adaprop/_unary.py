__all__ = ["BatchedReLUAdaPropNode"]

from src.cache import AdaPropCache
from src.nodes.batched_linprop import BatchedReLULinPropNode
from src.substorch import (
    ada_relax_relu,
    get_rlx_mask_relu,
    batched_ada_backsub_relu,
    batched_ada_backsub_relu_lower,
)


class BatchedReLUAdaPropNode(BatchedReLULinPropNode):
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
        backsub = (
            batched_ada_backsub_relu
            if len(symbnd) == 4
            else batched_ada_backsub_relu_lower
        )
        lower_ada_idx = self.cache.ada_rlx_idxs[self.name][0]
        lower_ada_param = self.cache.ada_rlx_prms[self.name][0]

        new_symbnd = backsub(
            *symbnd,
            *relaxation,
            lower_ada_idx=lower_ada_idx,
            lower_ada_param=lower_ada_param
        )
        self.cache.symbnds[self.name] = new_symbnd
