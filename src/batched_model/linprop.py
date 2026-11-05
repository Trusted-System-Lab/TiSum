__all__ = ["BatchedLinPropModel"]

from torch import Tensor

from src.args import BatchedLinPropArgs
from src.cache import LinPropCache
from src.enums import PropStatus
from src.model.linprop import LinPropModel
from src.model.utils import (
    split_coeffs_and_const,
    create_batched_or_groups,
    analyze_batched_each_prop_status,
)
from src.nodes import InputNode, BatchedBasicLinPropNode, OutputNode


class BatchedLinPropModel(LinPropModel):

    nodes: list[BatchedBasicLinPropNode]
    cache: LinPropCache

    _batch_size: int

    def __init__(
        self,
        nodes: list[BatchedBasicLinPropNode],
        arguments: BatchedLinPropArgs,
        verbose: bool = False,
    ):
        LinPropModel.__init__(self, nodes, arguments, verbose)
        self._batch_size = 1

    def set_batch_size(self, batch_size: int) -> None:
        self._batch_size = batch_size
        for node in self.nodes:
            node.set_batch_size(batch_size)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def set_input_constrs(self, input_constrs: Tensor):
        assert input_constrs.dim() == 3
        assert input_constrs.size(0) == self.batch_size
        assert input_constrs.size(2) == 2

        input_constrs = input_constrs.to(
            dtype=self.cache.dtype, device=self.cache.device
        )

        input_node: InputNode = self.nodes[0]
        shape = (-1, *input_node.output_sizes[0])
        self.cache.bnds[input_node.output_names[0]] = (
            input_constrs[:, :, 0].reshape(shape),
            input_constrs[:, :, 1].reshape(shape),
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
        or_groups = create_batched_or_groups(output_constrs, self.batch_size)

        output_node: OutputNode = self.nodes[-1]
        output_node.set_weight(weight)
        output_node.set_bias(bias)
        output_node.set_or_groups(or_groups)

        self._or_groups = or_groups
        self._each_prop_status = [[PropStatus.UNKNOWN] * len(g) for g in or_groups]

        return weight, bias

    def run(self, input_constrs: Tensor, output_constrs: list[Tensor]):
        self.set_input_constrs(input_constrs)
        self.set_output_constrs(output_constrs)

        self._run()

        output_bounds = self.cache.bnds[self.nodes[-1].name]

        self.each_prop_status = analyze_batched_each_prop_status(
            output_bounds, self.or_groups
        )

        if isinstance(self.cache, LinPropCache):
            if not self.arguments.store_output_final_symbnds:
                return

            assert not self.arguments.find_adv_example
