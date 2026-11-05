__all__ = ["BatchedMatMulCstLinPropNode"]

from src.nodes.basic import MatMulCstNode
from src.nodes.batched_linprop._basic import BatchedLinearLinPropNode
from src.utils import batched_store_and_update_bnd
from src.substorch import (
    batched_init_symbnd_matmulcst,
    batched_interval_matmulcst,
    batched_linear_backsub_matmulcst,
    batched_linear_backsub_matmulcst_lower,
)


class BatchedMatMulCstLinPropNode(MatMulCstNode, BatchedLinearLinPropNode):

    def _init_symbnd(self):
        """This method only initialize the symbolic bounds."""
        symbnd = batched_init_symbnd_matmulcst(self.weight, n_batches=self.batch_size)
        self._check_init_symbnd(symbnd)

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
        new_bound = batched_interval_matmulcst(
            *pre_bound, weight=self.weight, is_left_op=self.is_left_op
        )
        batched_store_and_update_bnd(new_bound, self.cache, self)

    def _bwdprop_symbnd(self):
        """This method back-substitutes the symbolic bounds by weights and biases."""
        symbnd = self._get_bwdprop_symbnd()
        backsub = (
            batched_linear_backsub_matmulcst
            if len(symbnd) == 4
            else batched_linear_backsub_matmulcst_lower
        )
        new_symbnd = backsub(*symbnd, weight=self.weight, is_left_op=self.is_left_op)
        self._check_bwdprop_symbnd(new_symbnd)
        self.cache.symbnds[self.name] = new_symbnd
