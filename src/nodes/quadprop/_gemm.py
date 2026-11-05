__all__ = ["GEMMQuadPropNode"]

from src.nodes.linprop import GEMMLinPropNode
from src.substorch import (
    linear_backsub_gemm,
    linear_backsub_gemm_lower,
    quad_backsub_gemm,
    quad_backsub_gemm_lower,
)


class GEMMQuadPropNode(GEMMLinPropNode):
    def _bwdprop_symbnd(self):
        pre_bound = self.cache.bnds[self.pre_nodes[0].name]
        pre_lower_bound, pre_upper_bound = pre_bound

        symbnd = self._get_bwdprop_symbnd()

        if len(symbnd) in {2, 4}:
            if len(symbnd) == 2:
                backsub = linear_backsub_gemm_lower
            else:
                backsub = linear_backsub_gemm
            new_symbnd = backsub(
                *symbnd,
                weight=self.weight,
                bias=self.bias,
            )
        elif len(symbnd) in {3, 6}:
            if len(symbnd) == 3:
                backsub = quad_backsub_gemm_lower
            else:
                backsub = quad_backsub_gemm
            new_symbnd = backsub(
                *symbnd,
                weight=self.weight,
                bias=self.bias,
                pre_lower_bound=pre_lower_bound,
                pre_upper_bound=pre_upper_bound,
            )
        else:
            raise ValueError(f"Unexpected symbolic bound length: {len(symbnd)}")

        self._check_bwdprop_symbnd(new_symbnd)
        self.cache.symbnds[self.name] = new_symbnd
