__all__ = ["BasicModel"]

from torch import Tensor

from propdag import TModel, clear_fwd_cache
from src.args import BasicArgs
from src.cache import BasicCache, LinPropCache
from src.enums import PropStatus, VerifStatus
from src.model.utils import (
    set_pre_next_nodes,
    split_coeffs_and_const,
    create_or_groups,
    analyze_each_prop_status,
    analyze_status,
    find_adv_example_by_final_symbnds,
)
from src.nodes import BasicNode, InputNode, OutputNode


class BasicModel(TModel):
    _nodes: list[BasicNode]
    _arguments: BasicArgs
    _all_backward_sorts: dict[BasicNode, list[BasicNode]]
    _cache: BasicCache
    _or_groups: list[list[int]] | None
    _each_prop_status: list[list[PropStatus]] | None
    _adv_example: tuple[Tensor, Tensor] | None = None

    arguments: BasicArgs
    nodes: list[BasicNode]
    cache: BasicCache
    or_groups: list[list[int]] | None
    each_prop_status: list[list[PropStatus]] | None
    adv_example: tuple[Tensor, Tensor] | None

    def __init__(
        self,
        nodes: list[BasicNode],
        arguments: BasicArgs,
        verbose: bool = False,
    ):
        set_pre_next_nodes(nodes)
        super().__init__(nodes, arguments.nodes_sort_strategy, verbose)
        assert isinstance(self.nodes[0], InputNode)
        assert isinstance(self.nodes[-1], OutputNode)

        self._arguments = arguments
        self._or_groups = None
        self._each_prop_status = None
        self._adv_example = None

    def set_input_constrs(self, input_constrs: Tensor):
        assert input_constrs.dim() == 2
        assert input_constrs.size(1) == 2

        input_constrs = input_constrs.to(
            dtype=self.cache.dtype, device=self.cache.device
        )

        input_node: InputNode = self.nodes[0]
        shape = input_node.output_sizes[0]
        self.cache.bnds[input_node.output_names[0]] = (
            input_constrs[:, 0].reshape(shape),
            input_constrs[:, 1].reshape(shape),
        )

    def set_output_constrs(self, output_constrs: list[Tensor]):
        assert all(c.dim() == output_constrs[0].dim() for c in output_constrs)
        assert all(c.size(1) == output_constrs[0].size(1) for c in output_constrs)

        output_constrs = [
            c.to(dtype=self.cache.dtype, device=self.cache.device)
            for c in output_constrs
        ]

        weight, bias = split_coeffs_and_const(output_constrs)
        weight, bias = (-weight).detach(), (-bias).detach()
        or_groups = create_or_groups(output_constrs)

        output_node: OutputNode = self.nodes[-1]
        output_node.set_weight(weight)
        output_node.set_bias(bias)
        output_node.set_or_groups(or_groups)

        self._or_groups = or_groups
        self.each_prop_status = [[PropStatus.UNKNOWN] * len(g) for g in or_groups]
        self._adv_example = None

        return weight, bias

    def analyze_status(self, output_bound: tuple[Tensor, ...]) -> VerifStatus:
        assert len(output_bound) == 2

        self.each_prop_status = analyze_each_prop_status(output_bound, self.or_groups)

        return analyze_status(self.each_prop_status)

    def _run(self):
        node = None
        cache_counter = {node: len(node.next_nodes) for node in self._nodes}
        for node in self.nodes:
            node.forward()
            if isinstance(node, InputNode):
                continue

            self.backsub(node)
            clear_fwd_cache(cache_counter, node.pre_nodes)
        clear_fwd_cache(cache_counter, [node])

    def run(self, input_constrs: Tensor, output_constrs: list[Tensor]):
        self.set_input_constrs(input_constrs)
        self.set_output_constrs(output_constrs)

        self._run()

        output_bounds = self.cache.bnds[self.nodes[-1].name]

        self.each_prop_status = analyze_each_prop_status(output_bounds, self.or_groups)

        if isinstance(self.cache, LinPropCache):
            if not self.arguments.store_output_final_symbnds:
                return

            if not self.arguments.find_adv_example:
                return

            status = analyze_status(self.each_prop_status)
            if status != VerifStatus.UNSAFE:
                return

            self._adv_example = find_adv_example_by_final_symbnds(
                self.cache.output_final_symbnds[0],
                self.cache.bnds[self.nodes[0].name],
                self.or_groups,
                self.arguments.onnx_model,
            )

    @property
    def or_groups(self):
        return self._or_groups

    @property
    def each_prop_status(self) -> list[list[PropStatus]] | None:
        return self._each_prop_status

    @each_prop_status.setter
    def each_prop_status(self, each_prop_status: list[list[PropStatus]] | None):
        if each_prop_status is None:
            self._each_prop_status = None
            return
        assert all(
            len(each_prop_status[i]) == len(self._or_groups[i])
            for i in range(len(self._or_groups))
        )
        self._each_prop_status = each_prop_status

    @property
    def adv_example(self):
        return self._adv_example

    @property
    def arguments(self) -> BasicArgs:
        return self._arguments
