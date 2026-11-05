__all__ = ["BatchedReLULinPropNode"]

from src.nodes.basic import ReLUNode
from src.nodes.batched_linprop._basic import (
    BatchedNonlinearLinPropNode,
    BatchedUnaryOpNode,
)
from src.utils import batched_store_and_update_bnd
from src.substorch import (
    relax_relu,
    interval_relu,
    batched_linear_backsub_unary,
    batched_linear_backsub_unary_lower,
)


class BatchedReLULinPropNode(ReLUNode, BatchedNonlinearLinPropNode, BatchedUnaryOpNode):

    def _build_rlx(self):
        assert len(self.pre_nodes) == 1

        pre_bound = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        relaxation = relax_relu(*pre_bound)
        self.cache.rlxs[self.name] = relaxation

    def _fwdprop_interval(self):
        assert len(self.pre_nodes) == 1

        pre_bound = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        new_bound = interval_relu(*pre_bound)
        batched_store_and_update_bnd(new_bound, self.cache, self)

    def _bwdprop_symbnd(self):
        symbnd = self._get_bwdprop_symbnd()
        relaxation = self.cache.rlxs[self.name]
        backsub = (
            batched_linear_backsub_unary
            if len(symbnd) == 4
            else (batched_linear_backsub_unary_lower)
        )
        new_symbnd = backsub(*symbnd, *relaxation)
        self.cache.symbnds[self.name] = new_symbnd
