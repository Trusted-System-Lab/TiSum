__all__ = ["Conv2dQuadPropNode"]

from torch import Tensor

from src.nodes.linprop import Conv2dLinPropNode
from src.substorch import (
    linear_backsub_conv2d,
    linear_backsub_conv2d_lower,
    quad_backsub_conv2d,
    quad_backsub_conv2d_lower,
)


class Conv2dQuadPropNode(Conv2dLinPropNode):
    input_sizes: tuple[tuple[int, int, int]]
    output_sizes: tuple[tuple[int, int, int]]
    output_padding: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    stride: tuple[int, int]
    groups: int
    weight: Tensor
    bias: Tensor

    def _bwdprop_symbnd(self):
        pre_bound = self.cache.bnds[self.pre_nodes[0].name]
        pre_lower_bound, pre_upper_bound = pre_bound

        symbnd = self._get_bwdprop_symbnd()

        if len(symbnd) in {2, 4}:
            if len(symbnd) == 2:
                backsub = linear_backsub_conv2d_lower
            else:
                backsub = linear_backsub_conv2d

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
        elif len(symbnd) in {3, 6}:

            if len(symbnd) == 3:
                backsub = quad_backsub_conv2d_lower
            else:
                backsub = quad_backsub_conv2d
            new_symbnd = backsub(
                *symbnd,
                weight=self.weight,
                bias=self.bias,
                input_size=self.input_sizes[0],
                output_size=self.output_sizes[0],
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                output_padding=self.output_padding,
                groups=self.groups,
                pre_lower_bound=pre_lower_bound,
                pre_upper_bound=pre_upper_bound,
            )
        else:
            raise ValueError(f"Unexpected symbolic bound length: {len(symbnd)}")

        self._check_bwdprop_symbnd(new_symbnd)
        self.cache.symbnds[self.name] = new_symbnd
