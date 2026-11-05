__all__ = ["BatchedGEMMLinPropNode"]

from src.nodes.basic import GEMMNode
from src.nodes.batched_linprop import BatchedLinearLinPropNode
from src.utils import batched_store_and_update_bnd
from src.substorch import (
    batched_init_symbnd_gemm,
    batched_interval_gemm,
    batched_linear_backsub_gemm,
    batched_linear_backsub_gemm_lower,
)


class BatchedGEMMLinPropNode(GEMMNode, BatchedLinearLinPropNode):

    def _init_symbnd(self):
        symbnd = batched_init_symbnd_gemm(
            self.weight, self.bias, n_batches=self.batch_size
        )
        symbnd_coeffs, symbnd_const = symbnd
        new_symbnd = (
            symbnd_coeffs,
            symbnd_const,
            symbnd_coeffs.clone(),
            symbnd_const.clone(),
        )
        self.cache.symbnds[self.name] = new_symbnd

    def _fwdprop_interval(self):
        assert len(self.pre_nodes) == 1

        pre_bound = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        new_bound = batched_interval_gemm(
            *pre_bound, weight=self.weight, bias=self.bias
        )
        batched_store_and_update_bnd(new_bound, self.cache, self)

    def _bwdprop_symbnd(self):
        symbnd = self._get_bwdprop_symbnd()
        backsub = (
            batched_linear_backsub_gemm
            if len(symbnd) == 4
            else batched_linear_backsub_gemm_lower
        )
        new_symbnd = backsub(*symbnd, weight=self.weight, bias=self.bias)
        self.cache.symbnds[self.name] = new_symbnd
