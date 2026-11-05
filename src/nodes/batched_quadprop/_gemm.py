__all__ = ["BatchedGEMMQuadPropNode"]

from src.nodes.batched_linprop import BatchedGEMMLinPropNode
from src.substorch import (
    batched_linear_backsub_gemm,
    batched_linear_backsub_gemm_lower,
    batched_quad_backsub_gemm,
    batched_quad_backsub_gemm_lower,
)


class BatchedGEMMQuadPropNode(BatchedGEMMLinPropNode):
    def _bwdprop_symbnd(self):
        pre_bound = self.cache.bnds[self.pre_nodes[0].name]
        pre_lower_bound, pre_upper_bound = pre_bound

        symbnd = self._get_bwdprop_symbnd()

        if len(symbnd) in {2, 4}:
            if len(symbnd) == 2:
                backsub = batched_linear_backsub_gemm_lower
            else:
                backsub = batched_linear_backsub_gemm
            new_symbnd = backsub(
                *symbnd,
                weight=self.weight,
                bias=self.bias,
            )
        elif len(symbnd) in {3, 6}:
            if len(symbnd) == 3:
                backsub = batched_quad_backsub_gemm_lower
            else:
                backsub = batched_quad_backsub_gemm
            new_symbnd = backsub(
                *symbnd,
                pre_lower_bound,
                pre_upper_bound,
                self.weight,
                self.bias,
            )
        else:
            raise ValueError(f"Unexpected symbolic bound length: {len(symbnd)}")

        self._check_bwdprop_symbnd(new_symbnd)
        self.cache.symbnds[self.name] = new_symbnd
