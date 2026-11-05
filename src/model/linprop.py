__all__ = ["LinPropModel"]

from propdag import clear_bwd_cache
from src.args import LinPropArgs
from src.cache import LinPropCache
from src.model.basic import BasicModel
from src.nodes import (
    BasicLinPropNode,
    LinearLinPropNode,
    NonlinearLinPropNode,
    ShapingLinPropNode,
    UnaryOpNode,
    SliceNode,
    ConcatNode,
    BinaryOpNode,
)


class LinPropModel(BasicModel):
    arguments: LinPropArgs
    cache: LinPropCache

    @staticmethod
    def _without_backsub(node: BasicLinPropNode):
        """The following cases can skip the backward pass: 1."""
        if (
            isinstance(node, (ShapingLinPropNode, SliceNode, ConcatNode))
            or (
                isinstance(node, UnaryOpNode)
                and not isinstance(node, NonlinearLinPropNode)
            )
            or (
                isinstance(node, NonlinearLinPropNode)
                and not (isinstance(node, BinaryOpNode) or node.pre_node_is_nonlinear)
            )
            or (isinstance(node, LinearLinPropNode) and node.pre_node_is_input)
        ):
            return True
        return False

    def _backsub(self, node: BasicLinPropNode):
        """Backsubstitute the symbolic bounds from the current node."""

        backward_sort = self._all_backward_sorts[node]
        cache_counter = {node: len(node.pre_nodes) for node in backward_sort}
        """
        All linear unary nodes can skip backward pass themselves by the weights and 
        bias.
        """
        if not isinstance(node, LinearLinPropNode) or isinstance(node, BinaryOpNode):
            node.backward()
        """
        Backward pass all preceding nodes.
        """
        for j in range(1, len(backward_sort)):
            node = backward_sort[j]
            node.backward()
            clear_bwd_cache(cache_counter, node.next_nodes)

        if (
            self.cache.cur_node == self.nodes[-1]
            and self.arguments.store_output_final_symbnds
        ):
            self.cache.output_final_symbnds = self.cache.symbnds[node.name]
        clear_bwd_cache(cache_counter, [node])

    def backsub(self, node: BasicLinPropNode):
        """Backsubstitute the symbolic bounds from the current node."""

        if self._without_backsub(node):
            return

        self._backsub(node)

        self.cache.cur_node = None
