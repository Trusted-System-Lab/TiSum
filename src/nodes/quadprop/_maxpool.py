__all__ = ["MaxPool2dQuadPropNode"]

from src.nodes.linprop import MaxPool2dLinPropNode
from src.nodes.quadprop._unary import ReLUQuadPropNode
from src.substorch import (
    quad_relax_maxpool2d_after_relu,
    quad_backsub_maxpool2d_after_relu,
    quad_backsub_maxpool2d_after_relu_lower,
)

from ._nonlinear import NonlinearQuadPropNode


class MaxPool2dQuadPropNode(MaxPool2dLinPropNode, NonlinearQuadPropNode):
    input_sizes: tuple[tuple[int, int, int]]
    output_sizes: tuple[tuple[int, int, int]]
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]

    def _cal_and_update_cur_node_bnd(self):
        return NonlinearQuadPropNode._cal_and_update_cur_node_bnd(self)

    def _build_rlx(self):
        assert len(self.pre_nodes) == 1, f"Invalid {len(self.pre_nodes)} pre-nodes."
        assert isinstance(
            self.pre_nodes[0], ReLUQuadPropNode
        ), f"Invalid pre-node type: {type(self.pre_nodes[0])}."

        pre_bound = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])

        (
            lower_relu_quad_coeffs,
            lower_relu_linear_coeffs,
            lower_relu_const,
            upper_relu_quad_coeffs,
            upper_relu_linear_coeffs,
            upper_relu_const,
        ) = self.cache.rlxs[self.pre_nodes[0].name]

        relaxation = quad_relax_maxpool2d_after_relu(
            *pre_bound,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            lower_relu_quad_coeffs=lower_relu_quad_coeffs,
            lower_relu_linear_coeffs=lower_relu_linear_coeffs,
            lower_relu_const=lower_relu_const,
            upper_relu_quad_coeffs=upper_relu_quad_coeffs,
            upper_relu_linear_coeffs=upper_relu_linear_coeffs,
            upper_relu_const=upper_relu_const,
        )
        self.cache.rlxs[self.name] = relaxation

    def _bwdprop_symbnd(self):
        symbnd = self._get_bwdprop_symbnd()
        relaxation = self.cache.rlxs[self.name]
        backsub = (
            quad_backsub_maxpool2d_after_relu
            if len(symbnd) == 4
            else quad_backsub_maxpool2d_after_relu_lower
        )
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
