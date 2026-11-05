__all__ = ["parse_node_infos"]

import onnx
import torch
from onnx import TensorProto, NodeProto, ValueInfoProto
from torch import Tensor

from src.args import BasicArgs
from src.containers import *
from src.slimonnx import get_input_nodes, get_output_nodes, get_initializers
from src.slimonnx.shapeonnx import infer_onnx_shape
from src.torchonnx import get_torch_args


def _to_tensor(initializer: TensorProto) -> Tensor:
    return torch.tensor(onnx.numpy_helper.to_array(initializer))


def _to_list(initializer: TensorProto) -> list | int:
    return onnx.numpy_helper.to_array(initializer).tolist()


def _filter_initers(
    names: tuple[str, ...],
    sizes: tuple[tuple[int, ...], ...],
    initializers: dict[str, TensorProto],
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    filtered_names = []
    filtered_sizes = []
    for name, size in zip(names, sizes):
        if name == "":
            filtered_names.append(name)
            filtered_sizes.append(())
        if name not in initializers:
            filtered_names.append(name)
            filtered_sizes.append(size)
    return tuple(filtered_names), tuple(filtered_sizes)


def _get_size(
    node: NodeProto | ValueInfoProto,
    data_shapes: dict[str, list[int]],
    remove_batch_dim: bool = True,
):
    def _get_sizes(node_names) -> tuple[tuple[int]]:
        sizes = []
        for node_name in node_names:
            if node_name == "":
                continue
            size = data_shapes.get(node_name)
            if size is None:
                raise RuntimeError(f"{node_name} not found in data shapes.")
            if remove_batch_dim:
                if len(size) != 1:
                    size = size[1:]
            size = tuple(size)
            sizes.append(size)
        sizes = tuple(sizes)
        return sizes

    input_sizes = _get_sizes(node.input)
    output_sizes = _get_sizes(node.output)

    return input_sizes, output_sizes


def _onnx2nodeinfo_unary(
    node: NodeProto,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    input_sizes: tuple[tuple[int, ...], ...],
    output_sizes: tuple[tuple[int, ...], ...],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
    type_name: str,
) -> AddCstNodeInfo | MatMulCstNodeInfo | None:
    if all(name in initializers for name in input_names):
        raise RuntimeError("Unary node taking two constants is not supported.")
    if len(input_names) < 2:
        raise RuntimeError(
            "Unary node should take one input or with an extra constant."
        )

    if input_names[0] in initializers:
        constant = _to_tensor(initializers[input_names[0]])
        is_left_op = True
    else:
        constant = _to_tensor(initializers[input_names[1]])
        is_left_op = False
    input_names, input_sizes = _filter_initers(input_names, input_sizes, initializers)

    if type_name == "Add":
        return AddCstNodeInfo(
            name=node.name,
            input_names=input_names,
            output_names=output_names,
            input_sizes=input_sizes,
            output_sizes=output_sizes,
            constant=constant,
        )
    elif type_name == "MatMul":
        return MatMulCstNodeInfo(
            name=node.name,
            input_names=input_names,
            output_names=output_names,
            input_sizes=input_sizes,
            output_sizes=output_sizes,
            weight=constant,
            is_left_op=is_left_op,
        )
    else:
        raise RuntimeError(f"Unknown type name {type_name}")


def _onnx2nodeinfo_binary(
    node: NodeProto,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    input_sizes: tuple[tuple[int, ...], ...],
    output_sizes: tuple[tuple[int, ...], ...],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
    type_name: str,
) -> AddNodeInfo | MatMulNodeInfo | None:
    if any(name in initializers for name in input_names):
        raise RuntimeError("Binary node taking constant as input is not supported.")
    if type_name == "Add":
        NodeInfo = AddNodeInfo
    elif type_name == "MatMul":
        NodeInfo = MatMulNodeInfo
    else:
        raise RuntimeError(f"Unknown type name {type_name}")
    return NodeInfo(
        name=node.name,
        input_names=input_names,
        output_names=output_names,
        input_sizes=input_sizes,
        output_sizes=output_sizes,
    )


def _onnx2nodeinfo_add(
    node: NodeProto,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    input_sizes: tuple[tuple[int, ...], ...],
    output_sizes: tuple[tuple[int, ...], ...],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> AddCstNodeInfo | AddNodeInfo:
    if any(name in initializers for name in input_names):
        return _onnx2nodeinfo_unary(
            node=node,
            input_names=input_names,
            output_names=output_names,
            input_sizes=input_sizes,
            output_sizes=output_sizes,
            nodes=nodes,
            initializers=initializers,
            type_name="Add",
        )
    return _onnx2nodeinfo_binary(
        node=node,
        input_names=input_names,
        output_names=output_names,
        input_sizes=input_sizes,
        output_sizes=output_sizes,
        nodes=nodes,
        initializers=initializers,
        type_name="Add",
    )


def _onnx2nodeinfo_concat(
    node: NodeProto,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    input_sizes: tuple[tuple[int, ...], ...],
    output_sizes: tuple[tuple[int, ...], ...],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> ConcatNodeInfo:
    constants = {}
    for name in input_names:
        if name in initializers:
            constants[name] = _to_tensor(initializers[name])
    attrs = get_torch_args(node, nodes, initializers)
    dim = attrs["dim"]
    dim = dim - 1 if dim > 0 else dim

    return ConcatNodeInfo(
        name=node.name,
        input_names=input_names,
        output_names=output_names,
        input_sizes=input_sizes,
        output_sizes=output_sizes,
        dim=dim,
        constants=constants,
    )


def _onnx2nodeinfo_conv(
    node: NodeProto,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    input_sizes: tuple[tuple[int, ...], ...],
    output_sizes: tuple[tuple[int, ...], ...],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> ConvNodeInfo:
    attrs = get_torch_args(node, nodes, initializers)
    kernel_size = attrs["kernel_size"]
    if len(kernel_size) != 2:
        raise RuntimeError(f"Only 2D convolution is supported, but got {kernel_size}.")

    padding = attrs["padding"]
    dilation = attrs["dilation"]
    stride = attrs["stride"]

    if type(padding) is not int:
        raise RuntimeError(
            "Only 1D padding is supported, that is each side has the same padding."
        )
    if type(dilation) is not int:
        raise RuntimeError(
            "Only 1D dilation is supported, that is each side has the same dilation."
        )
    if type(stride) is not int:
        raise RuntimeError(
            "Only 1D stride is supported, that is each side has the same stride."
        )

    padding = tuple([padding] * 2)
    dilation = tuple([dilation] * 2)
    stride = tuple([stride] * 2)
    groups = attrs["groups"]
    weight = _to_tensor(initializers[input_names[1]])
    bias = (
        _to_tensor(initializers[input_names[2]])
        if len(input_names) == 3
        else torch.zeros(weight.shape[0])
    )

    input_names, input_sizes = _filter_initers(input_names, input_sizes, initializers)
    return Conv2dNodeInfo(
        name=node.name,
        input_names=input_names,
        output_names=output_names,
        input_sizes=input_sizes,
        output_sizes=output_sizes,
        kernel_size=kernel_size,
        padding=padding,
        dilation=dilation,
        stride=stride,
        groups=groups,
        weight=weight,
        bias=bias,
    )


def _onnx2nodeinfo_flatten(
    node: NodeProto,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    input_sizes: tuple[tuple[int, ...], ...],
    output_sizes: tuple[tuple[int, ...], ...],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> FlattenNodeInfo:
    attrs = get_torch_args(node, nodes, initializers)
    start_dim = attrs["start_dim"]
    start_dim = (
        start_dim - 1 if start_dim > 0 else start_dim
    )

    return FlattenNodeInfo(
        name=node.name,
        input_names=input_names,
        output_names=output_names,
        input_sizes=input_sizes,
        output_sizes=output_sizes,
        start_dim=start_dim,
    )


def _onnx2nodeinfo_gemm(
    node: NodeProto,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    input_sizes: tuple[tuple[int, ...], ...],
    output_sizes: tuple[tuple[int, ...], ...],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> GEMMNodeInfo:
    attrs = get_torch_args(node, nodes, initializers)
    assert attrs["alpha"] == 1.0
    assert attrs["beta"] == 1.0
    assert attrs["transA"] == 0
    assert attrs["transB"] == 0

    weight = _to_tensor(initializers[input_names[1]])
    bias = (
        _to_tensor(initializers[input_names[2]])
        if len(input_names) == 3
        else torch.zeros(weight.shape[0])
    )

    input_names, input_sizes = _filter_initers(input_names, input_sizes, initializers)
    return GEMMNodeInfo(
        name=node.name,
        input_names=input_names,
        output_names=output_names,
        input_sizes=input_sizes,
        output_sizes=output_sizes,
        weight=weight,
        bias=bias,
    )


def _onnx2nodeinfo_matmul(
    node: NodeProto,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    input_sizes: tuple[tuple[int, ...], ...],
    output_sizes: tuple[tuple[int, ...], ...],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> MatMulCstNodeInfo | MatMulNodeInfo:
    if any(name in initializers for name in input_names):
        return _onnx2nodeinfo_unary(
            node=node,
            input_names=input_names,
            output_names=output_names,
            input_sizes=input_sizes,
            output_sizes=output_sizes,
            nodes=nodes,
            initializers=initializers,
            type_name="MatMul",
        )
    return _onnx2nodeinfo_binary(
        node=node,
        input_names=input_names,
        output_names=output_names,
        input_sizes=input_sizes,
        output_sizes=output_sizes,
        nodes=nodes,
        initializers=initializers,
        type_name="MatMul",
    )


def _onnx2nodeinfo_maxpool(
    node: NodeProto,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    input_sizes: tuple[tuple[int, ...], ...],
    output_sizes: tuple[tuple[int, ...], ...],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> MaxPoolNodeInfo:
    attrs = get_torch_args(node, nodes, initializers)
    kernel_size = attrs["kernel_size"]
    if len(kernel_size) != 2:
        raise RuntimeError("Only 2D average pooling is supported.")

    padding = attrs["padding"]
    dilation = attrs["dilation"]
    stride = attrs["stride"]

    if type(padding) is not int:
        raise RuntimeError(
            "Only 1D padding is supported, that is each side has the same padding."
        )
    if type(dilation) is not int:
        raise RuntimeError(
            "Only 1D dilation is supported, that is each side has the same dilation."
        )
    if type(stride) is not int:
        raise RuntimeError(
            "Only 1D stride is supported, that is each side has the same stride."
        )

    padding = tuple([padding] * 2)
    dilation = tuple([dilation] * 2)
    stride = tuple([stride] * 2)
    ceil_mode = attrs["ceil_mode"]
    return_indices = attrs["return_indices"]

    input_names, input_sizes = _filter_initers(input_names, input_sizes, initializers)
    return MaxPool2dNodeInfo(
        name=node.name,
        input_names=input_names,
        output_names=output_names,
        input_sizes=input_sizes,
        output_sizes=output_sizes,
        kernel_size=kernel_size,
        padding=padding,
        dilation=dilation,
        stride=stride,
        ceil_mode=ceil_mode,
        return_indices=return_indices,
    )


def _onnx2nodeinfo_relu(
    node: NodeProto,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    input_sizes: tuple[tuple[int, ...], ...],
    output_sizes: tuple[tuple[int, ...], ...],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> ReLUNodeInfo:
    return ReLUNodeInfo(
        name=node.name,
        input_names=input_names,
        output_names=output_names,
        input_sizes=input_sizes,
        output_sizes=output_sizes,
    )


def _onnx2nodeinfo_reshape(
    node: NodeProto,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    input_sizes: tuple[tuple[int, ...], ...],
    output_sizes: tuple[tuple[int, ...], ...],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> ReshapeNodeInfo:
    shape = None
    if input_names[1] in initializers:
        shape = output_sizes[0]

    input_names, input_sizes = _filter_initers(input_names, input_sizes, initializers)
    return ReshapeNodeInfo(
        name=node.name,
        input_names=input_names,
        output_names=output_names,
        input_sizes=input_sizes,
        output_sizes=output_sizes,
        shape=shape,
    )


def _onnx2nodeinfo_slice(
    node: NodeProto,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    input_sizes: tuple[tuple[int, ...], ...],
    output_sizes: tuple[tuple[int, ...], ...],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> SliceNodeInfo:
    if input_names[1] in initializers:
        starts = _to_list(initializers[input_names[1]])
    else:
        raise NotImplementedError(
            "Slice node without starts as an initializer is not supported."
        )
    if input_names[2] in initializers:
        ends = _to_list(initializers[input_names[2]])
    else:
        raise NotImplementedError(
            "Slice node without ends as an initializer is not supported."
        )
    if input_names[3] in initializers:
        axes = _to_list(initializers[input_names[3]])
        new_axes = []
        for axis in axes:
            if axis == 0:
                raise RuntimeError("Slice node with batch dimension is not supported.")
            elif axis > 0:
                new_axes.append(axis - 1)
            else:
                new_axes.append(axis)
        axes = new_axes
    else:
        raise NotImplementedError(
            "Slice node without axes as an initializer is not supported."
        )
    if input_names[4] in initializers:
        steps = _to_list(initializers[input_names[4]])
    else:
        steps = [1] * len(starts)

    input_names, input_sizes = _filter_initers(input_names, input_sizes, initializers)
    return SliceNodeInfo(
        name=node.name,
        input_names=input_names,
        output_names=output_names,
        input_sizes=input_sizes,
        output_sizes=output_sizes,
        starts=starts,
        ends=ends,
        axes=axes,
        steps=steps,
    )


def _onnx2nodeinfo_sigmoid(
    node: NodeProto,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    input_sizes: tuple[tuple[int, ...], ...],
    output_sizes: tuple[tuple[int, ...], ...],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> SigmoidNodeInfo:
    return SigmoidNodeInfo(
        name=node.name,
        input_names=input_names,
        output_names=output_names,
        input_sizes=input_sizes,
        output_sizes=output_sizes,
    )


def _onnx2nodeinfo_tanh(
    node: NodeProto,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    input_sizes: tuple[tuple[int, ...], ...],
    output_sizes: tuple[tuple[int, ...], ...],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> TanhNodeInfo:
    return TanhNodeInfo(
        name=node.name,
        input_names=input_names,
        output_names=output_names,
        input_sizes=input_sizes,
        output_sizes=output_sizes,
    )


def _onnx2nodeinfo_input(
    node: ValueInfoProto,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    input_sizes: tuple[tuple[int, ...], ...],
    output_sizes: tuple[tuple[int, ...], ...],
) -> InputNodeInfo:
    return InputNodeInfo(
        name=node.name,
        input_names=input_names,
        output_names=output_names,
        input_sizes=input_sizes,
        output_sizes=output_sizes,
    )


def _onnx2nodeinfo_output(
    node: ValueInfoProto,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    input_sizes: tuple[tuple[int, ...], ...],
    output_sizes: tuple[tuple[int, ...], ...],
) -> OutputNodeInfo:
    return OutputNodeInfo(
        name=node.name,
        input_names=input_names,
        output_names=output_names,
        input_sizes=input_sizes,
        output_sizes=output_sizes,
    )


_NODE_INFO_MAP = {
    "Add": _onnx2nodeinfo_add,
    "Concat": _onnx2nodeinfo_concat,
    "Conv": _onnx2nodeinfo_conv,
    "Flatten": _onnx2nodeinfo_flatten,
    "Gemm": _onnx2nodeinfo_gemm,
    "MatMul": _onnx2nodeinfo_matmul,
    "MaxPool": _onnx2nodeinfo_maxpool,
    "Relu": _onnx2nodeinfo_relu,
    "Reshape": _onnx2nodeinfo_reshape,
    "Sigmoid": _onnx2nodeinfo_sigmoid,
    "Slice": _onnx2nodeinfo_slice,
    "Tanh": _onnx2nodeinfo_tanh,
}


def parse_node_infos(arguments: BasicArgs) -> list[BasicNodeInfo]:
    onnx_file_path = arguments.slimonnx_path
    model = onnx.load(onnx_file_path)

    input_nodes = get_input_nodes(model)
    output_nodes = get_output_nodes(model)
    initializers = get_initializers(model)
    nodes = list(model.graph.node)
    data_shapes = infer_onnx_shape(input_nodes, output_nodes, nodes, initializers)

    nodes_infos = []
    for node in input_nodes:
        input_names = ()
        output_names = (node.name,)
        input_sizes = ()
        output_sizes = (tuple(data_shapes[node.name][1:]),)
        node_info = _onnx2nodeinfo_input(
            node=node,
            input_names=input_names,
            output_names=output_names,
            input_sizes=input_sizes,
            output_sizes=output_sizes,
        )
        nodes_infos.append(node_info)

    nodes_dict = {node.name: node for node in nodes}
    for node in nodes:
        input_names = tuple([name for name in node.input])
        output_names = tuple([name for name in node.output])
        input_sizes, output_sizes = _get_size(node, data_shapes)
        node_info = _NODE_INFO_MAP[node.op_type](
            node=node,
            input_names=input_names,
            output_names=output_names,
            input_sizes=input_sizes,
            output_sizes=output_sizes,
            nodes=nodes_dict,
            initializers=initializers,
        )
        nodes_infos.append(node_info)

    for node in output_nodes:
        input_names = (node.name,)
        output_names = ()
        input_sizes = (tuple(data_shapes[node.name][1:]),)
        output_sizes = ()
        node_info = _onnx2nodeinfo_output(
            node=node,
            input_names=input_names,
            output_names=output_names,
            input_sizes=input_sizes,
            output_sizes=output_sizes,
        )
        nodes_infos.append(node_info)

    return nodes_infos
