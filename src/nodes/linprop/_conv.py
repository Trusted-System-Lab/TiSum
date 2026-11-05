__all__ = ["Conv2dLinPropNode"]

from torch import Tensor

from src.nodes.basic import Conv2dNode
from src.nodes.linprop._basic import LinearLinPropNode
from src.utils import store_and_update_bnd
from src.substorch import (
    init_symbnd_conv2d,
    interval_conv2d,
    backsub_conv2d,
    backsub_conv2d_lower,
)


class Conv2dLinPropNode(Conv2dNode, LinearLinPropNode):
    output_sizes: tuple[tuple[int, int, int]]
    output_padding: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    stride: tuple[int, int]
    groups: int
    weight: Tensor
    bias: Tensor

    def _init_symbnd(self):
        """This method only initialize the symbolic bounds."""
        symbnd = init_symbnd_conv2d(
            self.weight,
            self.bias,
            output_size=self.output_sizes[0],
            padding=self.padding,
            dilation=self.dilation,
            stride=self.stride,
            output_padding=self.output_padding,
            groups=self.groups,
        )
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
        new_bound = interval_conv2d(
            *pre_bound,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        store_and_update_bnd(new_bound, self.cache, self)

    def _bwdprop_symbnd(self):
        """This method back-substitutes the symbolic bounds by weights and biases."""

        symbnd = self._get_bwdprop_symbnd()

        backsub = backsub_conv2d if len(symbnd) == 4 else backsub_conv2d_lower
        new_symbnd = backsub(
            *symbnd,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            output_padding=self.output_padding,
            groups=self.groups,
        )
        self._check_bwdprop_symbnd(new_symbnd)

        self.cache.symbnds[self.name] = new_symbnd
