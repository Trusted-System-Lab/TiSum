__all__ = ["MaxPool2dAdaPropNode"]

from src.cache import AdaPropCache
from src.nodes.linprop import MaxPool2dLinPropNode
from src.substorch import (
    ada_relax_maxpool2d,
    get_rlx_mask_maxpool2d,
    cal_l_argmax_maxpool2d,
    unfold_pre_bound_maxpool2d,
    ada_backsub_maxpool2d,
    ada_backsub_maxpool2d_lower,
)


class MaxPool2dAdaPropNode(MaxPool2dLinPropNode):
    cache: AdaPropCache
    input_sizes: tuple[tuple[int, int, int]]
    output_sizes: tuple[tuple[int, int, int]]
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]

    def _build_rlx(self):
        assert len(self.pre_nodes) == 1

        pre_bound = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        pre_lower_bound, pre_upper_bound = pre_bound
        output_size = self.output_sizes[0]
        unfold_pre_lower_bound, unfold_pre_upper_bound = unfold_pre_bound_maxpool2d(
            pre_lower_bound,
            pre_upper_bound,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        pre_lower_bound_max, pre_lower_bound_argmax = cal_l_argmax_maxpool2d(
            unfold_pre_lower_bound, unfold_pre_upper_bound
        )
        ada_mask = get_rlx_mask_maxpool2d(
            unfold_pre_lower_bound,
            unfold_pre_upper_bound,
            pre_lower_bound_max,
            pre_lower_bound_argmax,
        )
        relaxation, lower_ada_idx, lower_ada_param = ada_relax_maxpool2d(
            *pre_bound,
            ada_mask=ada_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        self.cache.rlxs[self.name] = relaxation
        self.cache.ada_rlx_idxs[self.name] = (lower_ada_idx, None)
        self.cache.ada_rlx_prms[self.name] = (lower_ada_param, None)

    def _bwdprop_symbnd(self):
        symbnd = self._get_bwdprop_symbnd()
        relaxation = self.cache.rlxs[self.name]
        backsub = (
            ada_backsub_maxpool2d if len(symbnd) == 4 else ada_backsub_maxpool2d_lower
        )
        lower_ada_idx = self.cache.ada_rlx_idxs[self.name][0]
        lower_ada_param = self.cache.ada_rlx_prms[self.name][0]
        new_symbnd = backsub(
            *symbnd,
            *relaxation,
            input_size=self.input_sizes[0],
            output_size=self.output_sizes[0],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            lower_ada_idx=lower_ada_idx,
            lower_ada_param=lower_ada_param,
        )
        self._check_bwdprop_symbnd(new_symbnd)

        self.cache.symbnds[self.name] = new_symbnd
