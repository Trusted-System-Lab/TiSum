__all__ = [
    "ReLUQuadPropNode",
    "SigmoidQuadPropNode",
    "TanhQuadPropNode",
]

from src.nodes.basic import MaxPool2dNode
from src.nodes.linprop import ReLULinPropNode, SigmoidLinPropNode, TanhLinPropNode
from src.nodes.quadprop._nonlinear import NonlinearQuadPropNode
from src.substorch import (
    RelaxMode,
    quad_relax_relu,
    quad_relax_sigmoid,
    quad_relax_tanh,
    quad_backsub_unary,
    quad_backsub_unary_lower,
)


class ReLUQuadPropNode(ReLULinPropNode, NonlinearQuadPropNode):

    def _cal_and_update_cur_node_bnd(self):
        return NonlinearQuadPropNode._cal_and_update_cur_node_bnd(self)

    def _build_rlx(self):
        mode = self.argument.act_relax_mode
        assert mode == RelaxMode.SINGLE_QUAD, f"Invalid relaxation mode: {mode}."
        assert len(self.pre_nodes) == 1, f"Invalid {len(self.pre_nodes)} pre-nodes."

        pre_bound = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        use_lower_quad_rlx = isinstance(self.next_nodes[0], MaxPool2dNode)
        relaxation = quad_relax_relu(*pre_bound, use_lower_quad_rlx=use_lower_quad_rlx)
        self.cache.rlxs[self.name] = relaxation

    def _bwdprop_symbnd(self):
        symbnd = self._get_bwdprop_symbnd()
        if isinstance(self.next_nodes[0], MaxPool2dNode) and len(symbnd) in {3, 6}:
            self.cache.symbnds[self.name] = symbnd
            return

        relaxation = self.cache.rlxs[self.name]
        new_symbnd = (
            quad_backsub_unary(*symbnd, *relaxation)
            if len(symbnd) == 4
            else quad_backsub_unary_lower(*symbnd, *relaxation)
        )
        self.cache.symbnds[self.name] = new_symbnd


class SigmoidQuadPropNode(SigmoidLinPropNode, NonlinearQuadPropNode):

    def _cal_and_update_cur_node_bnd(self):
        return NonlinearQuadPropNode._cal_and_update_cur_node_bnd(self)

    def _build_rlx(self):
        mode = self.argument.act_relax_mode
        assert mode == RelaxMode.SINGLE_QUAD, f"Invalid relaxation mode: {mode}."
        assert len(self.pre_nodes) == 1, f"Invalid {len(self.pre_nodes)} pre-nodes."

        pre_bound = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        relaxation = quad_relax_sigmoid(*pre_bound)
        self.cache.rlxs[self.name] = relaxation

    def _bwdprop_symbnd(self):
        symbnd = self._get_bwdprop_symbnd()
        relaxation = self.cache.rlxs[self.name]
        new_symbnd = (
            quad_backsub_unary(*symbnd, *relaxation)
            if len(symbnd) == 4
            else quad_backsub_unary_lower(*symbnd, *relaxation)
        )
        self.cache.symbnds[self.name] = new_symbnd


class TanhQuadPropNode(TanhLinPropNode, NonlinearQuadPropNode):

    def _cal_and_update_cur_node_bnd(self):
        return NonlinearQuadPropNode._cal_and_update_cur_node_bnd(self)

    def _build_rlx(self):
        mode = self.argument.act_relax_mode
        assert mode == RelaxMode.SINGLE_QUAD, f"Invalid relaxation mode: {mode}."
        assert len(self.pre_nodes) == 1, f"Invalid {len(self.pre_nodes)} pre-nodes."

        pre_bound = self._get_fwdprop_bnd(self.pre_nodes[0], self.input_names[0])
        relaxation = quad_relax_tanh(*pre_bound)
        self.cache.rlxs[self.name] = relaxation

    def _bwdprop_symbnd(self):
        symbnd = self._get_bwdprop_symbnd()
        relaxation = self.cache.rlxs[self.name]
        new_symbnd = (
            quad_backsub_unary(*symbnd, *relaxation)
            if len(symbnd) == 4
            else quad_backsub_unary_lower(*symbnd, *relaxation)
        )
        self.cache.symbnds[self.name] = new_symbnd
