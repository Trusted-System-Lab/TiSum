__all__ = [
    "BasicNode",
    "PoolNode",
    "AddNode",
    "AddCstNode",
    "ConcatNode",
    "ConvNode",
    "Conv2dNode",
    "FlattenNode",
    "GEMMNode",
    "MatMulNode",
    "MatMulCstNode",
    "MaxPoolNode",
    "MaxPool2dNode",
    "ReLUNode",
    "ReshapeNode",
    "SigmoidNode",
    "SliceNode",
    "TanhNode",
    "InputNode",
    "OutputNode",
]

from typing import ClassVar

import torch
from torch import Tensor

from propdag import TNode
from src.args import BasicArgs
from src.cache import BasicCache
from src.containers import *


class BasicNode(TNode):
    _name: str
    _input_names: tuple[str, ...]
    _output_names: tuple[str, ...]
    _input_sizes: tuple[tuple[int, ...], ...]
    _output_sizes: tuple[tuple[int, ...], ...]
    _cache: BasicCache
    _argument: BasicArgs
    _node_info: BasicNodeInfo | None
    _pre_nodes: list["BasicNode"]
    _next_nodes: list["BasicNode"]
    _dtype: torch.dtype
    _device: torch.device

    def __init__(
        self, node_info: BasicNodeInfo, cache: BasicCache, argument: BasicArgs
    ):
        TNode.__init__(self, node_info.name, cache, argument)
        self._dtype = cache.dtype
        self._device = cache.device
        self._unpack_node_info(node_info)

    def _unpack_node_info(self, node_info: BasicNodeInfo):
        self._name = node_info.name
        self._input_names = node_info.input_names
        self._output_names = node_info.output_names
        self._input_sizes = node_info.input_sizes
        self._output_sizes = node_info.output_sizes

    @property
    def input_names(self) -> tuple[str, ...]:
        return self._input_names

    @property
    def output_names(self):
        return self._output_names

    @property
    def input_sizes(self) -> tuple[tuple[int, ...], ...]:
        return self._input_sizes

    @property
    def output_sizes(self) -> tuple[tuple[int, ...], ...]:
        return self._output_sizes

    @property
    def cache(self) -> BasicCache:
        return self._cache

    @cache.setter
    def cache(self, value: BasicCache):
        self._cache = value

    @property
    def argument(self) -> BasicArgs:
        return self._argument

    @argument.setter
    def argument(self, value: BasicArgs):
        self._argument = value

    @property
    def pre_nodes(self) -> list["BasicNode"]:
        return self._pre_nodes

    @pre_nodes.setter
    def pre_nodes(self, value: list["BasicNode"]):
        self._pre_nodes = value

    @property
    def next_nodes(self) -> list["BasicNode"]:
        return self._next_nodes

    @next_nodes.setter
    def next_nodes(self, value: list["BasicNode"]):
        self._next_nodes = value

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device


class PoolNode(BasicNode):
    _kernel_size: tuple[int, ...]
    _padding: tuple[int, ...]
    _dilation: tuple[int, ...]
    _stride: tuple[int, ...]

    def _unpack_node_info(self, node_info: PoolNodeInfo):
        super()._unpack_node_info(node_info)
        self._kernel_size = node_info.kernel_size
        self._padding = node_info.padding
        self._dilation = node_info.dilation
        self._stride = node_info.stride

    @property
    def kernel_size(self) -> tuple[int, ...]:
        return self._kernel_size

    @property
    def padding(self) -> tuple[int, ...]:
        return self._padding

    @property
    def dilation(self) -> tuple[int, ...]:
        return self._dilation

    @property
    def stride(self) -> tuple[int, ...]:
        return self._stride


class AddNode(BasicNode):
    pass


class AddCstNode(BasicNode):
    _constant: Tensor

    def _unpack_node_info(self, node_info: AddCstNodeInfo):
        super()._unpack_node_info(node_info)
        self._constant = node_info.constant.to(dtype=self.dtype, device=self.device)

    @property
    def constant(self) -> Tensor:
        return self._constant


class ConcatNode(BasicNode):
    _dim: int
    _constants: dict[str, Tensor]

    def _unpack_node_info(self, node_info: ConcatNodeInfo):
        super()._unpack_node_info(node_info)
        self._dim = node_info.dim
        self._constants = node_info.constants
        for key, value in self._constants.items():
            self._constants[key] = value.to(dtype=self.dtype, device=self.device)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def constants(self) -> dict[str, Tensor]:
        return self._constants


class ConvNode(PoolNode):
    _groups: int
    _output_padding: tuple[int, ...]
    _weight: Tensor
    _bias: Tensor

    def _unpack_node_info(self, node_info: ConvNodeInfo):
        super()._unpack_node_info(node_info)
        self._groups = node_info.groups
        self._output_padding = node_info.output_padding
        self._weight = node_info.weight.to(dtype=self.dtype, device=self.device)
        self._bias = node_info.bias.to(dtype=self.dtype, device=self.device)

    @property
    def groups(self) -> int:
        return self._groups

    @property
    def output_padding(self) -> tuple[int, ...]:
        return self._output_padding

    @property
    def weight(self) -> Tensor:
        return self._weight

    @property
    def bias(self) -> Tensor:
        return self._bias


class Conv2dNode(ConvNode):
    pass


class FlattenNode(BasicNode):
    _start_dim: int

    def _unpack_node_info(self, node_info: FlattenNodeInfo):
        super()._unpack_node_info(node_info)
        self._start_dim = node_info.start_dim

    @property
    def start_dim(self) -> int:
        return self._start_dim


class GEMMNode(BasicNode):
    _weight: Tensor
    _bias: Tensor

    def _unpack_node_info(self, node_info: GEMMNodeInfo):
        super()._unpack_node_info(node_info)
        self._weight = node_info.weight.to(dtype=self.dtype, device=self.device)
        self._bias = node_info.bias.to(dtype=self.dtype, device=self.device)

    @property
    def weight(self) -> Tensor:
        return self._weight

    @property
    def bias(self) -> Tensor:
        return self._bias


class MatMulNode(BasicNode):
    pass


class MatMulCstNode(BasicNode):
    _weight: Tensor
    _is_left_op: bool

    def _unpack_node_info(self, node_info: MatMulCstNodeInfo):
        super()._unpack_node_info(node_info)
        self._weight = node_info.weight.to(dtype=self.dtype, device=self.device)
        self._is_left_op = node_info.is_left_op

    @property
    def weight(self) -> Tensor:
        return self._weight

    @property
    def is_left_op(self):
        return self._is_left_op


class MaxPoolNode(PoolNode):
    _ceil_mode: bool
    _output_padding: tuple[int, ...]
    _return_indices: bool

    def _unpack_node_info(self, node_info: MaxPoolNodeInfo):
        super()._unpack_node_info(node_info)
        self._ceil_mode = node_info.ceil_mode
        self._output_padding = node_info.output_padding
        self._return_indices = node_info.return_indices

    @property
    def ceil_mode(self):
        return self._ceil_mode

    @property
    def output_padding(self):
        return self._output_padding

    @property
    def return_indices(self):
        return self._return_indices


class MaxPool2dNode(MaxPoolNode):
    pass


class ReLUNode(BasicNode):
    act_type: ClassVar[str] = "relu"


class ReshapeNode(BasicNode):
    _shape: tuple[int, ...]

    def _unpack_node_info(self, node_info: ReshapeNodeInfo):
        super()._unpack_node_info(node_info)
        self._shape = node_info.shape

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape


class SigmoidNode(BasicNode):
    act_type: ClassVar[str] = "sigmoid"


class SliceNode(BasicNode):
    _starts: list[int]
    _ends: list[int]
    _steps: list[int]
    _axes: list[int]

    def _unpack_node_info(self, node_info: SliceNodeInfo):
        super()._unpack_node_info(node_info)
        self._starts = node_info.starts
        self._ends = node_info.ends
        self._steps = node_info.steps
        self._axes = node_info.axes

    @property
    def starts(self) -> list[int] | None:
        return self._starts

    @property
    def ends(self) -> list[int] | None:
        return self._ends

    @property
    def steps(self) -> list[int] | None:
        return self._steps

    @property
    def axes(self) -> list[int] | None:
        return self._axes


class TanhNode(BasicNode):
    act_type: ClassVar[str] = "tanh"


class InputNode(BasicNode):

    def forward(self):
        if len(self.pre_nodes) == 0:
            assert self.name in self.cache.bnds
            return


class OutputNode(BasicNode):
    _weight: Tensor | None
    _bias: Tensor | None
    _or_groups: list[list[int]] | None

    @property
    def weight(self) -> Tensor | None:
        return self._weight.to(dtype=self.dtype, device=self.device)

    @property
    def bias(self) -> Tensor | None:
        return self._bias.to(dtype=self.dtype, device=self.device)

    @property
    def or_groups(self) -> list[list[int]] | None:
        return self._or_groups

    def set_weight(self, weight: Tensor):
        self._weight = weight.t().to(dtype=self.dtype, device=self.device).detach()

    def set_bias(self, bias: Tensor):
        self._bias = bias.to(dtype=self.dtype, device=self.device).detach()

    def set_or_groups(self, or_groups: list[list[int]] | None):
        self._or_groups = or_groups
