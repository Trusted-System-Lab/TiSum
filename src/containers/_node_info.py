__all__ = [
    "BasicNodeInfo",
    "UnaryOpNodeInfo",
    "BinaryOpNodeInfo",
    "PoolNodeInfo",
    "AddCstNodeInfo",
    "AddNodeInfo",
    "ConcatNodeInfo",
    "ConvNodeInfo",
    "Conv2dNodeInfo",
    "FlattenNodeInfo",
    "GEMMNodeInfo",
    "MatMulCstNodeInfo",
    "MatMulNodeInfo",
    "MaxPoolNodeInfo",
    "MaxPool2dNodeInfo",
    "ReLUNodeInfo",
    "ReshapeNodeInfo",
    "SigmoidNodeInfo",
    "SliceNodeInfo",
    "TanhNodeInfo",
    "InputNodeInfo",
    "OutputNodeInfo",
]

import math
from abc import ABC
from dataclasses import dataclass, field
from typing import ClassVar

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class BasicNodeInfo(ABC):
    name: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    input_sizes: tuple[tuple[int, ...], ...]
    output_sizes: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class UnaryOpNodeInfo(BasicNodeInfo):
    pass


@dataclass(frozen=True, slots=True)
class BinaryOpNodeInfo(BasicNodeInfo):
    pass


@dataclass(frozen=True, slots=True)
class PoolNodeInfo(BasicNodeInfo):
    kernel_size: tuple[int, ...]
    padding: tuple[int, ...]
    dilation: tuple[int, ...]
    stride: tuple[int, ...]

    def _cal_output_data_size(self) -> tuple[list[int], list[int]]:
        input_size = self.input_sizes[0][1:]
        d = len(input_size)

        output_size = [0] * d
        output_padding = [0] * d
        for i in range(d):
            t1 = 2 * self.padding[i]
            t2 = self.dilation[i] * (self.kernel_size[i] - 1)
            t3 = input_size[i] + t1 - t2 - 1
            size_f = t3 / self.stride[1] + 1
            if hasattr(self, "ceil_mode") and getattr(self, "ceil_mode") is True:
                output_size_i = math.ceil(size_f)
            else:
                output_size_i = math.floor(size_f)
            output_size[i] = output_size_i
            output_padding[i] = t3 - (output_size_i - 1) * self.stride[i]

        return output_size, output_padding

    def _set_output_padding(self):
        output_data_size, output_padding = self._cal_output_data_size()

        output_padding = tuple(output_padding)
        if hasattr(self, "output_padding") and self.output_padding is not None:
            assert self.output_padding == output_padding
        object.__setattr__(self, "output_padding", output_padding)

    def _get_output_channel(self):
        return self.output_sizes[0][0]


@dataclass(frozen=True, slots=True)
class AddNodeInfo(BinaryOpNodeInfo):
    pass


@dataclass(frozen=True, slots=True)
class AddCstNodeInfo(UnaryOpNodeInfo):
    constant: Tensor = field(repr=False)


@dataclass(frozen=True, slots=True)
class ConcatNodeInfo(BasicNodeInfo):
    dim: int = field(default=0)
    constants: dict[str, Tensor] = field(default_factory=dict, repr=False)


@dataclass(frozen=True, slots=True)
class ConvNodeInfo(PoolNodeInfo):
    weight: Tensor = field(repr=False)
    bias: Tensor = field(repr=False)
    groups: int = field(default=1)
    output_padding: tuple[int, ...] = field(default=None)

    def __post_init__(self):
        self._set_output_padding()

    def _get_output_channel(self):
        return self.weight.shape[0]


@dataclass(frozen=True, slots=True)
class Conv2dNodeInfo(ConvNodeInfo):

    def __post_init__(self):
        ConvNodeInfo.__post_init__(self)
        assert self.weight.dim() == 4
        assert self.bias.dim() == 1
        assert self.weight.shape[0] == self.output_sizes[0][0]
        assert self.weight.shape[1] == int(self.input_sizes[0][0] / self.groups)
        assert self.weight.shape[2:] == self.kernel_size
        assert self.bias.shape[0] == self.output_sizes[0][0]
        assert len(self.kernel_size) == 2
        assert len(self.padding) == 2
        assert len(self.dilation) == 2
        assert len(self.stride) == 2


@dataclass(frozen=True, slots=True)
class FlattenNodeInfo(UnaryOpNodeInfo):
    start_dim: int = field(default=1)


@dataclass(frozen=True, slots=True)
class GEMMNodeInfo(BasicNodeInfo):
    weight: Tensor = field(repr=False)
    bias: Tensor = field(repr=False)

    def __post_init__(self):
        assert self.weight.dim() == 2, f"Weight must be 2D, got {self.weight.dim()}D."
        assert self.bias.dim() == 1, f"Bias must be 1D, got {self.bias.dim()}D."
        assert self.weight.shape[0] == self.input_sizes[0][-1], (
            f"Input size {self.input_sizes[0]} does not match weight shape "
            f"{self.weight.shape}."
        )
        assert self.weight.shape[1] == self.output_sizes[0][-1], (
            f"Output size {self.output_sizes[0]} does not match weight shape "
            f"{self.weight.shape}."
        )
        assert self.bias.shape[0] == self.output_sizes[0][-1], (
            f"Output size {self.output_sizes[0]} does not match bias shape "
            f"{self.bias.shape}."
        )


@dataclass(frozen=True, slots=True)
class MatMulNodeInfo(BinaryOpNodeInfo):
    pass


@dataclass(frozen=True, slots=True)
class MatMulCstNodeInfo(BinaryOpNodeInfo):
    weight: Tensor = field(repr=False)
    is_left_op: bool = field(default=True)


@dataclass(frozen=True, slots=True)
class MaxPoolNodeInfo(PoolNodeInfo):
    ceil_mode: bool = field(default=False)
    return_indices: bool = field(default=False)
    output_padding: tuple[int, ...] = field(default=None)

    def __post_init__(self):
        self._set_output_padding()
        assert self.return_indices is False


@dataclass(frozen=True, slots=True)
class MaxPool2dNodeInfo(MaxPoolNodeInfo):

    def __post_init__(self):
        MaxPoolNodeInfo.__post_init__(self)
        assert len(self.kernel_size) == 2
        assert len(self.padding) == 2
        assert len(self.dilation) == 2
        assert len(self.stride) == 2


@dataclass(frozen=True, slots=True)
class ReLUNodeInfo(UnaryOpNodeInfo):
    act_type: ClassVar[str] = "relu"


@dataclass(frozen=True, slots=True)
class ReshapeNodeInfo(UnaryOpNodeInfo):
    shape: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class SigmoidNodeInfo(UnaryOpNodeInfo):
    act_type: ClassVar[str] = "sigmoid"


@dataclass(frozen=True, slots=True)
class SliceNodeInfo(BasicNodeInfo):
    starts: list[int]
    ends: list[int]
    axes: list[int]
    steps: list[int]


@dataclass(frozen=True, slots=True)
class TanhNodeInfo(UnaryOpNodeInfo):
    act_type: ClassVar[str] = "tanh"


@dataclass(frozen=True, slots=True)
class InputNodeInfo(BasicNodeInfo):
    pass


@dataclass(frozen=True, slots=True)
class OutputNodeInfo(BasicNodeInfo):
    pass
