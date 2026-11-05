__all__ = [
    "BatchedBasicLinPropNode",
    "BatchedUnaryOpNode",
    "BatchedBinaryOpNode",
    "BatchedShapingLinPropNode",
    "BatchedLinearLinPropNode",
    "BatchedNonlinearLinPropNode",
]

from torch import Tensor

from src.args import BatchedLinPropArgs
from src.cache import LinPropCache
from src.nodes.basic import InputNode
from src.nodes.linprop import (
    BasicLinPropNode,
    BinaryOpNode,
    LinearLinPropNode,
    NonlinearLinPropNode,
    ShapingLinPropNode,
    UnaryOpNode,
)
from src.utils import batched_store_and_update_bnd
from src.substorch import (
    batched_cal_scalar_bound,
    batched_cal_scalar_bound_lower,
    batched_init_symbnd_similar_eye,
)


class BatchedBasicLinPropNode(BasicLinPropNode):
    _cache: LinPropCache
    _argument: BatchedLinPropArgs
    _pre_nodes: list["BatchedBasicLinPropNode"]
    _next_nodes: list["BatchedBasicLinPropNode"]
    _batch_size: int

    def set_batch_size(self, batch_size: int):
        self._batch_size = batch_size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _cal_and_update_cur_node_bnd(self):
        """This method calculate the scalar bounds based on the input bounds."""
        symbnd = self.cache.symbnds[self.name]
        bound = self.cache.bnds[self.name]
        new_bound = (
            batched_cal_scalar_bound(*symbnd, *bound)
            if len(symbnd) == 4
            else (batched_cal_scalar_bound_lower(*symbnd, *bound),)
        )

        batched_store_and_update_bnd(new_bound, self.cache, self.cache.cur_node)

    def _init_symbnd(self):
        symbnd_coeffs, symbnd_const = batched_init_symbnd_similar_eye(
            self.output_sizes[0],
            n_batches=self.batch_size,
            dtype=self.dtype,
            device=self.device,
        )

        symbnd = (
            symbnd_coeffs,
            symbnd_const,
            symbnd_coeffs.clone(),
            symbnd_const.clone(),
        )

        self.cache.symbnds[self.name] = symbnd

    def _get_fwdprop_bnd(
        self, pre_node: "BasicLinPropNode", input_name: str
    ) -> tuple[Tensor, ...]:
        """The method is to obtain the bounds (input_name) from the preceding node (pre_node)."""
        pre_bound = self.cache.bnds[pre_node.name]
        if isinstance(pre_bound[0], tuple):
            raise NotImplementedError(
                "BatchedBasicLinPropNode does not support multiple pre nodes."
            )

        return pre_bound

    def _get_bwdprop_symbnd(
        self,
    ) -> tuple[Tensor, ...]:
        def _get_symbnd_from_one_next_node(
            output_name: str, next_node: "BasicLinPropNode"
        ) -> tuple[Tensor, ...] | None:
            symbnd = self.cache.symbnds.get(next_node.name)
            if symbnd is None:
                return None

            if not isinstance(symbnd[0], tuple):
                if len(next_node.input_names) == 1:
                    return symbnd

                n_next_inputs = len(next_node.input_names)
                length_symbnd = len(symbnd)
                length_single_symbnd = length_symbnd // n_next_inputs
                for i, input_name in enumerate(next_node.input_names):
                    if input_name == output_name:
                        new_lower_symbnd = symbnd[2 * i : 2 * i + 2]
                        if length_single_symbnd == 2:
                            return new_lower_symbnd
                        if length_single_symbnd == 4:
                            new_upper_symbnd = symbnd[
                                2 * i
                                + length_single_symbnd : 2 * i
                                + length_single_symbnd
                                + 2
                            ]
                            return (*new_lower_symbnd, *new_upper_symbnd)
                        raise RuntimeError(
                            f"Invalid length of symbolic bounds: "
                            f"{length_single_symbnd}."
                        )
                raise RuntimeError(
                    f"Cannot find the symbolic bounds for "
                    f"{output_name} in {next_node.input_names}."
                )

            symbnd_tuple: tuple = symbnd
            for i, input_name in enumerate(next_node.input_names):
                if input_name == output_name:
                    symbnd = symbnd_tuple[i]
                    return symbnd
            raise RuntimeError(
                f"Cannot find the symbolic bounds for "
                f"{output_name} in {next_node.input_names}."
            )

        def _sum_symbnds_from_multiple_next_nodes(
            output_name: str,
            next_nodes_group: list["BasicLinPropNode"] | tuple["BasicLinPropNode", ...],
        ) -> tuple[Tensor, ...] | None:
            symbnd = None
            for next_node in next_nodes_group:
                temp_symbnd = _get_symbnd_from_one_next_node(output_name, next_node)
                if temp_symbnd is None:
                    continue
                if symbnd is None:
                    symbnd = [temp.clone() for temp in temp_symbnd]
                    continue

                assert len(symbnd) == len(temp_symbnd)
                symbnd[0] += temp_symbnd[0]
                symbnd[1] += temp_symbnd[1]
                if len(temp_symbnd) == 4:
                    symbnd[2] += temp_symbnd[2]
                    symbnd[3] += temp_symbnd[3]

            if symbnd is not None:
                return tuple(symbnd)
            return None

        if len(self.output_names) == 1:
            if len(self.next_nodes) == 1:
                symbnd = _get_symbnd_from_one_next_node(
                    self.output_names[0], self.next_nodes[0]
                )
            else:
                symbnd = _sum_symbnds_from_multiple_next_nodes(
                    self.output_names[0], self.next_nodes
                )

            if symbnd is None:
                symbnd = self.cache.symbnds[self.name]
            return symbnd
        raise NotImplementedError(
            "BatchedBasicLinPropNode does not support multiple output names."
        )
        output_name_to_next_nodes_group = {
            output_name: [] for output_name in self.output_names
        }
        for output_name in self.output_names:
            for next_node in self.next_nodes:
                if output_name in next_node.input_names:
                    output_name_to_next_nodes_group[output_name].append(next_node)

        symbnd_list: list[tuple[Tensor, ...] | None]
        symbnd_list = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            symbnd_list[i] = _sum_symbnds_from_multiple_next_nodes(
                output_name, output_name_to_next_nodes_group[output_name]
            )

        assert any(symbnd_list is not None)

        is_only_symbnds = False
        n_batch = -1
        for symbnd in symbnd_list:
            if len(symbnd) == 2:
                is_only_symbnds = True
                n_batch = symbnd[0].shape[0]
                break

        assert n_batch != -1

        dtype, device = self.dtype, self.device
        kwargs = {"dtype": dtype, "device": device}
        for i in range(len(symbnd_list)):
            symbnd = symbnd_list[i]
            if symbnd is not None:
                continue

            shape = self.input_sizes[i]
            symbnd = [
                torch.zeros((n_batch, *shape), **kwargs),
                torch.zeros((n_batch,), **kwargs),
            ]
            if not is_only_symbnds:
                symbnd.append(symbnd[0].clone())
                symbnd.append(symbnd[1].clone())
            symbnd = tuple(symbnd)
            symbnd_list[i] = symbnd

        assert all(symbnd is not None for symbnd in symbnd_list)

        return tuple(symbnd_list)

    def _check_init_symbnd(self, symbnd: tuple[Tensor, ...]):
        pass

    def _check_bwdprop_symbnd(self, symbnd: tuple[Tensor, ...]):
        pass


class BatchedUnaryOpNode(BatchedBasicLinPropNode, UnaryOpNode):
    constant: Tensor

    def _reduce_broadcast_after_backsub(
        self, symbnd: tuple[Tensor, ...]
    ) -> tuple[Tensor, ...]:
        raise NotImplementedError(
            f"_reduce_broadcast_after_backsub is not implemented in "
            f"{type(self).__name__}."
        )


class BatchedBinaryOpNode(BatchedBasicLinPropNode, BinaryOpNode):

    def _reduce_broadcast_after_backsub(
        self, symbnd: tuple[Tensor, ...]
    ) -> tuple[Tensor, ...]:
        raise NotImplementedError(
            f"_reduce_broadcast_after_backsub is not implemented in "
            f"{type(self).__name__}."
        )


class BatchedShapingLinPropNode(BatchedBasicLinPropNode, ShapingLinPropNode):
    pass


class BatchedLinearLinPropNode(BatchedBasicLinPropNode, LinearLinPropNode):
    _pre_node_is_input: bool
    pre_node_is_input: bool

    def forward(self):
        BatchedBasicLinPropNode.forward(self)
        if isinstance(self, BatchedUnaryOpNode) or self.pre_node_is_input:
            """
            If the node is a unary linear operation, we can directly calculate the
            bounds by interval arithmetic without losing precision obtained from
            back-substitute symbolic bounds.

            If the preceding node is an input node, we can also directly calculate the
            bounds by interval arithmetic without losing precision obtained from
            back-substitute symbolic bounds. This is to save memory and time.
            """
            self._fwdprop_interval()
        else:
            """
            We need initialize the symbolic bounds for back-substitution. For all linear
            nodes, we can skip the backward pass by directly using the coeffs and bias
            of the linear operation. But this is only for calculate their bounds in the
            common cases.
            """
            if self.argument.refine_by_interval_arithmetic:
                """
                If we refine the symbolic bounds by interval arithmetic, we need to
                calculate the bounds by interval arithmetic first.
                """
                self._fwdprop_interval()

            self._init_symbnd()

    @property
    def pre_node_is_input(self) -> bool:
        """This property is lazy initialized because we need to make sure the pre_nodes and next_nodes are set before we check the pre_node_is_input."""
        if not hasattr(self, "_pre_node_is_input"):
            setattr(
                self,
                "_pre_node_is_input",
                len(self.pre_nodes) == 1 and isinstance(self.pre_nodes[0], InputNode),
            )
        return self._pre_node_is_input


class BatchedNonlinearLinPropNode(BatchedBasicLinPropNode, NonlinearLinPropNode):
    _pre_node_is_nonlinear: bool
    _in_single_path: bool
    pre_node_is_nonlinear: bool
    in_single_path: bool

    def forward(self):
        BatchedBasicLinPropNode.forward(self)
        """
        For nonlinear nodes, the first step is to calculate the relaxation based on the
        preceding node's bounds.
        """
        self._build_rlx()
        """
        Commonly, we can only calculate the bounds by interval arithmetic without losing
        precision obtained from back-substituting symbolic bounds.

        However, if the preceding node is a nonlinear node, we need to back-substitute
        the symbolic bounds to obtain a tighter bound.

        Here, we do not use if-else statement and always process the interval 
        arithmetic to avoid a precision loss. Because the results of the 
        back-substitution may not be better than the interval arithmetic in some cases.
        """
        self._fwdprop_interval()

        if self.pre_node_is_nonlinear or isinstance(self, BatchedBinaryOpNode):
            """
            Use the back-substituted symbolic bound to calculate the bound.
            """
            self._init_symbnd()

    def backward(self, store_bnd: bool = True):
        if self.argument.refine_by_scalarize_each_layer and self.in_single_path:
            """
            Because the nonlinear node's bounds are obtained by interval arithmetic
            wherein the interval arithmetic is based on the output domain of the
            nonlinear node. So the results of the interval arithmetic may not be worse
            than the back-substituting symbolic bounds.

            So sometimes, we can use this way to get a tighter bounds.

            For a general DAG, we need to check if the current node is in a single path.
            If not, we do not support back-substituting the symbolic bounds because
            it maybe very hard to consider the soundness.
            """
            if store_bnd:
                self._cal_and_update_cur_node_bnd()

        BatchedBasicLinPropNode.backward(self)

    def _build_rlx(self):
        raise RuntimeError(
            f"This methods should be instantiated in {type(self).__name__}."
        )

    @property
    def pre_node_is_nonlinear(self) -> bool:
        """This property is lazy initialized because we need to make sure the pre_nodes and next_nodes are set before we check the pre_node_is_nonlinear."""
        if not hasattr(self, "_pre_node_is_nonlinear"):
            setattr(
                self,
                "_pre_node_is_nonlinear",
                len(self.pre_nodes) == 1
                and isinstance(self.pre_nodes[0], BatchedNonlinearLinPropNode),
            )
        return self._pre_node_is_nonlinear
