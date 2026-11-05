__all__ = ["MatMulLinPropNode", "MatMulCstLinPropNode"]

from src.nodes.basic import MatMulNode, MatMulCstNode
from src.nodes.linprop._basic import (
    LinearLinPropNode,
    NonlinearLinPropNode,
    BinaryOpNode,
)
from src.utils import store_and_update_bnd
from src.substorch import (
    relax_matmul,
    init_symbnd_matmulcst,
    interval_matmul,
    interval_matmulcst,
    backsub_matmul,
    backsub_matmul_lower,
    backsub_matmulcst,
    backsub_matmulcst_lower,
)


class MatMulLinPropNode(MatMulNode, NonlinearLinPropNode, BinaryOpNode):

    def _build_rlx(self):
        assert len(self.pre_nodes) == 2

        pre_bound_1 = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        pre_bound_2 = self._get_fwdprop_bnd(self.pre_nodes[1], self.input_names[1])
        relaxation = relax_matmul(*pre_bound_1, *pre_bound_2)
        self.cache.rlxs[self.name] = relaxation

    def _fwdprop_interval(self):
        assert len(self.pre_nodes) == 2
        pre_bound_1 = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        pre_bound_2 = self._get_fwdprop_bnd(self.pre_nodes[1], self.input_names[1])
        new_bound = interval_matmul(*pre_bound_1, *pre_bound_2)
        store_and_update_bnd(new_bound, self.cache, self)

    def _bwdprop_symbnd(self):
        symbnd = self._get_bwdprop_symbnd()
        relaxation = self.cache.rlxs[self.name]
        backsub = backsub_matmul if len(symbnd) == 4 else backsub_matmul_lower
        new_symbnd = backsub(*symbnd, *relaxation)
        new_symbnd = self._reduce_broadcast_after_backsub(new_symbnd)
        self.cache.symbnds[self.name] = new_symbnd


class MatMulCstLinPropNode(MatMulCstNode, LinearLinPropNode):

    def _init_symbnd(self):
        symbnd = init_symbnd_matmulcst(self.weight)
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
        new_bound = interval_matmulcst(
            *pre_bound, weight=self.weight, is_left_op=self.is_left_op
        )
        store_and_update_bnd(new_bound, self.cache, self)

    def _bwdprop_symbnd(self):
        symbnd = self._get_bwdprop_symbnd()
        backsub = backsub_matmulcst if len(symbnd) == 4 else backsub_matmulcst_lower
        new_symbnd = backsub(*symbnd, weight=self.weight, is_left_op=self.is_left_op)
        self._check_bwdprop_symbnd(new_symbnd)
        self.cache.symbnds[self.name] = new_symbnd
