__all__ = [
    "BasicLinPropNode",
    "UnaryOpNode",
    "BinaryOpNode",
    "ShapingLinPropNode",
    "LinearLinPropNode",
    "NonlinearLinPropNode",
]

import torch
from torch import Tensor

from src.args import LinPropArgs
from src.cache import LinPropCache
from src.nodes.basic import BasicNode, InputNode
from src.utils import store_and_update_bnd
from src.substorch import cal_scalar_bound, cal_scalar_bound_lower, init_symbnd_similar_eye

"""
The following classes should not have an __init__ method but like an interface.
"""


class BasicLinPropNode(BasicNode):
    _cache: LinPropCache
    _argument: LinPropArgs
    _pre_nodes: list["BasicLinPropNode"]
    _next_nodes: list["BasicLinPropNode"]

    def forward(self):
        self.cache.cur_node = self

    def backward(self, store_bnd: bool = True):
        self._bwdprop_symbnd()

    def _fwdprop_interval(self):
        raise RuntimeError(
            f"This methods should be instantiated in {type(self).__name__}."
        )

    def _fwdprop_symbnd(self):
        raise RuntimeError(f"Forward pass is not supported in backward mode.")

    def _cal_and_update_cur_node_bnd(self):
        """This method calculate the scalar bounds based on the input bounds."""
        symbnd = self.cache.symbnds[self.name]
        bound = self.cache.bnds[self.name]
        new_bound = (
            cal_scalar_bound(*symbnd, *bound)
            if len(symbnd) == 4
            else (cal_scalar_bound_lower(*symbnd, *bound),)
        )

        store_and_update_bnd(new_bound, self.cache, self.cache.cur_node)

    def _init_symbnd(self):
        """The default method of initializing symbolic bounds is to generate a similar eye matrix but with output size."""

        symbnd_coeffs, symbnd_const = init_symbnd_similar_eye(
            self.output_sizes[0], dtype=self.dtype, device=self.device
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
            for i, output_name in enumerate(self.pre_nodes[0].output_names):
                if input_name == output_name:
                    pre_bound = pre_bound[i]
                    break

        return pre_bound

    def _get_bwdprop_symbnd(
        self,
    ) -> tuple[Tensor, ...] | tuple[tuple[Tensor, ...]]:
        """This method is to obtain the symbolic bounds in current back-substitution."""

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
        if len(symbnd) in {2, 4}:
            assert (
                symbnd[0].shape[1:] == self.input_sizes[0]
            ), f"{symbnd[0].shape} != {self.input_sizes[0]}"
            assert symbnd[1].dim() == 1, f"{symbnd[1].dim()}"
            assert (
                symbnd[1].shape[0] == symbnd[0].shape[0]
            ), f"{symbnd[1].shape} != {symbnd[0].shape[0]}"
            if len(symbnd) == 4:
                assert (
                    symbnd[2].shape[1:] == self.input_sizes[0]
                ), f"{symbnd[2].shape} != {self.input_sizes[0]}"
                assert symbnd[3].dim() == 1, f"{symbnd[3].dim()}"
                assert (
                    symbnd[3].shape[0] == symbnd[0].shape[0]
                ), f"{symbnd[3].shape} != {symbnd[0].shape[0]}"

    def clear_fwd_cache(self):
        if self.argument.store_all_bnds or self.argument.refine_by_scalarize_each_layer:
            """
            We need store bounds of all layers sometimes.

            We need keep the bounds if we set the refine_by_scalarize_each_layer=True.
            We need the bounds to update the bounds when back-substituting the symbolic
            bounds.
            """
            return
        if len(self.next_nodes) > 0 and len(self.pre_nodes) > 0:
            """
            We need keep the bounds of the input and output nodes.
            """
            del self.cache.bnds[self.name]

    def clear_bwd_cache(self):
        del self.cache.symbnds[self.name]

    @property
    def cache(self) -> LinPropCache:
        return self._cache

    @cache.setter
    def cache(self, value: LinPropCache):
        self._cache = value

    @property
    def argument(self) -> LinPropArgs:
        return self._argument

    @argument.setter
    def argument(self, value: LinPropArgs):
        self._argument = value

    @property
    def pre_nodes(self) -> list["BasicLinPropNode"]:
        return self._pre_nodes

    @pre_nodes.setter
    def pre_nodes(self, value: list["BasicLinPropNode"]):
        self._pre_nodes = value

    @property
    def next_nodes(self) -> list["BasicLinPropNode"]:
        return self._next_nodes

    @next_nodes.setter
    def next_nodes(self, value: list["BasicLinPropNode"]):
        self._next_nodes = value


"""
The following classes are auxiliary classes for choosing different methods.
"""


def _reduce_broadcast(symbnd: Tensor, reduced_shape: tuple[int, ...]) -> Tensor:
    assert len(symbnd.shape) - 1 == len(
        reduced_shape
    ), f"{symbnd.shape}， {reduced_shape}"

    reduced_dims = []
    for i in range(len(symbnd.shape) - 1):
        if symbnd.shape[i + 1] != reduced_shape[i]:
            reduced_dims.append(i + 1)
    symbnd = torch.sum(symbnd, dim=reduced_dims, keepdim=True)
    return symbnd


class UnaryOpNode(BasicLinPropNode):
    constant: Tensor

    def _reduce_broadcast_after_backsub(
        self, symbnd: tuple[Tensor, ...]
    ) -> tuple[Tensor, ...]:
        symbnd = list(symbnd)
        input_shape = self.input_sizes[0]
        assert len(symbnd) in {2, 4}, f"{len(symbnd)}"
        for i in range(0, len(symbnd), 2):
            if symbnd[i].shape[1:] != input_shape:
                symbnd[i] = _reduce_broadcast(symbnd[i], input_shape)

        return tuple(symbnd)


class BinaryOpNode(BasicLinPropNode):

    def _reduce_broadcast_after_backsub(
        self, symbnd: tuple[Tensor, ...]
    ) -> tuple[Tensor, ...]:
        symbnd = list(symbnd)
        input_shape1, input_shape2 = self.input_sizes
        assert len(symbnd) in {4, 8}, f"{len(symbnd)}"
        for i in range(0, len(symbnd), 4):
            if symbnd[i].shape[1:] != input_shape1:
                symbnd[i] = _reduce_broadcast(symbnd[i], input_shape1)
            if symbnd[i + 2].shape[1:] != input_shape2:
                symbnd[i + 2] = _reduce_broadcast(symbnd[i + 2], input_shape2)

        return tuple(symbnd)


class ShapingLinPropNode(BasicLinPropNode):
    pass


class LinearLinPropNode(BasicLinPropNode):
    _pre_node_is_input: bool
    pre_node_is_input: bool

    def forward(self):
        BasicLinPropNode.forward(self)
        if isinstance(self, UnaryOpNode) or self.pre_node_is_input:
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


class NonlinearLinPropNode(BasicLinPropNode):
    _pre_node_is_nonlinear: bool
    _in_single_path: bool
    pre_node_is_nonlinear: bool
    in_single_path: bool

    def forward(self):
        BasicLinPropNode.forward(self)
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

        if self.pre_node_is_nonlinear or isinstance(self, BinaryOpNode):
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

        BasicLinPropNode.backward(self)

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
                and isinstance(self.pre_nodes[0], NonlinearLinPropNode),
            )
        return self._pre_node_is_nonlinear

    @property
    def in_single_path(self) -> bool:
        """This property is lazy initialized because we need to make sure the pre_nodes and next_nodes are set before we check the in_single_path."""
        return len(self.cache.symbnds) == 2
