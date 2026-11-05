__all__ = ["GEMMLinPropNode"]

from src.nodes.basic import GEMMNode
from src.nodes.linprop._basic import LinearLinPropNode
from src.utils import store_and_update_bnd
from src.substorch import init_symbnd_gemm, interval_gemm, backsub_gemm, backsub_gemm_lower


class GEMMLinPropNode(GEMMNode, LinearLinPropNode):

    def _init_symbnd(self):
        """This method only initialize the symbolic bounds."""
        symbnd = init_symbnd_gemm(self.weight, self.bias)
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
        new_bound = interval_gemm(*pre_bound, weight=self.weight, bias=self.bias)
        store_and_update_bnd(new_bound, self.cache, self)

    def _bwdprop_symbnd(self):
        """This method back-substitutes the symbolic bounds by weights and biases."""
        symbnd = self._get_bwdprop_symbnd()
        backsub = backsub_gemm if len(symbnd) == 4 else backsub_gemm_lower
        new_symbnd = backsub(*symbnd, weight=self.weight, bias=self.bias)
        self._check_bwdprop_symbnd(new_symbnd)
        self.cache.symbnds[self.name] = new_symbnd
