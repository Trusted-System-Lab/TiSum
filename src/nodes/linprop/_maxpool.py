__all__ = ["MaxPool2dLinPropNode"]

from src.nodes.basic import MaxPool2dNode
from src.nodes.linprop._basic import NonlinearLinPropNode
from src.utils import store_and_update_bnd
from src.substorch import (
    relax_maxpool2d,
    interval_maxpool2d,
    backsub_maxpool2d,
    backsub_maxpool2d_lower,
)


class MaxPool2dLinPropNode(MaxPool2dNode, NonlinearLinPropNode):
    input_sizes: tuple[tuple[int, int, int]]
    output_sizes: tuple[tuple[int, int, int]]
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]

    def _build_rlx(self):
        assert len(self.pre_nodes) == 1

        pre_bound = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        relaxation = relax_maxpool2d(
            *pre_bound,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        self.cache.rlxs[self.name] = relaxation

    def _fwdprop_interval(self):
        assert len(self.pre_nodes) == 1

        pre_bound = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        new_bound = interval_maxpool2d(
            *pre_bound,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
        )
        store_and_update_bnd(new_bound, self.cache, self)

    def _bwdprop_symbnd(self):
        symbnd = self._get_bwdprop_symbnd()
        relaxation = self.cache.rlxs[self.name]
        backsub = backsub_maxpool2d if len(symbnd) == 4 else backsub_maxpool2d_lower
        new_symbnd = backsub(
            *symbnd,
            *relaxation,
            input_size=self.input_sizes[0],
            output_size=self.output_sizes[0],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        self._check_bwdprop_symbnd(new_symbnd)

        self.cache.symbnds[self.name] = new_symbnd
