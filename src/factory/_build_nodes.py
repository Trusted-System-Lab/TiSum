__all__ = ["build_nodes_by_infos"]

import types

import torch

from src.args import *
from src.cache import *
from src.containers import *
from src.nodes import *

_CACHE_MAP = types.MappingProxyType(
    {
        LinPropArgs: LinPropCache,
        BatchedLinPropArgs: LinPropCache,
        AdaPropArgs: AdaPropCache,
        BatchedAdaPropArgs: AdaPropCache,
        QuadPropArgs: QuadPropCache,
        BatchedQuadPropArgs: QuadPropCache,
        LinProgArgs: LinProgCache,
        InputSplitPropArgs: BaBPropCache,
        BatchedInputSplitPropArgs: BaBPropCache,
    }
)

_LINPROP_NODES = types.MappingProxyType(
    {
        AddNodeInfo: AddLinPropNode,
        AddCstNodeInfo: AddCstLinPropNode,
        ConcatNodeInfo: ConcatLinPropNode,
        Conv2dNodeInfo: Conv2dLinPropNode,
        FlattenNodeInfo: FlattenLinPropNode,
        GEMMNodeInfo: GEMMLinPropNode,
        MatMulCstNodeInfo: MatMulCstLinPropNode,
        MaxPool2dNodeInfo: MaxPool2dLinPropNode,
        MatMulNodeInfo: MatMulLinPropNode,
        ReLUNodeInfo: ReLULinPropNode,
        ReshapeNodeInfo: ReshapeLinPropNode,
        SliceNodeInfo: SliceLinPropNode,
        SigmoidNodeInfo: SigmoidLinPropNode,
        TanhNodeInfo: TanhLinPropNode,
        InputNodeInfo: InputLinPropNode,
        OutputNodeInfo: OutputLinPropNode,
    }
)
_BATCHED_LINPROP_NODES = types.MappingProxyType(
    {
        FlattenNodeInfo: BatchedFlattenLinPropNode,
        GEMMNodeInfo: BatchedGEMMLinPropNode,
        InputNodeInfo: BatchedInputLinPropNode,
        ReLUNodeInfo: BatchedReLULinPropNode,
        OutputNodeInfo: BatchedOutputLinPropNode,
        MatMulCstNodeInfo: BatchedMatMulCstLinPropNode,
        ConcatNodeInfo: BatchedConcatLinPropNode,
        SliceNodeInfo: BatchedSliceLinPropNode,
    }
)
_ADAPROP_NODES = types.MappingProxyType(
    {
        Conv2dNodeInfo: Conv2dAdaPropNode,
        FlattenNodeInfo: FlattenAdaPropNode,
        GEMMNodeInfo: GEMMAdaPropNode,
        InputNodeInfo: InputAdaPropNode,
        MaxPool2dNodeInfo: MaxPool2dAdaPropNode,
        OutputNodeInfo: OutputAdaPropNode,
        ReLUNodeInfo: ReLUAdaPropNode,
        SigmoidNodeInfo: SigmoidAdaPropNode,
        TanhNodeInfo: TanhAdaPropNode,
        MatMulCstNodeInfo: MatMulCstAdaPropNode,
        ConcatNodeInfo: ConcatAdaPropNode,
        SliceNodeInfo: SliceAdaPropNode,
    }
)
_BATCHED_ADAPROP_NODES = types.MappingProxyType(
    {
        FlattenNodeInfo: BatchedFlattenAdaPropNode,
        GEMMNodeInfo: BatchedGEMMAdaPropNode,
        InputNodeInfo: BatchedInputAdaPropNode,
        ReLUNodeInfo: BatchedReLUAdaPropNode,
        OutputNodeInfo: BatchedOutputAdaPropNode,
        MatMulCstNodeInfo: BatchedMatMulCstAdaPropNode,
        ConcatNodeInfo: BatchedConcatAdaPropNode,
        SliceNodeInfo: BatchedSliceAdaPropNode,
    }
)
_QUADPROP_NODES = types.MappingProxyType(
    {
        Conv2dNodeInfo: Conv2dQuadPropNode,
        FlattenNodeInfo: FlattenQuadPropNode,
        GEMMNodeInfo: GEMMQuadPropNode,
        InputNodeInfo: InputQuadPropNode,
        MaxPool2dNodeInfo: MaxPool2dQuadPropNode,
        OutputNodeInfo: OutputQuadPropNode,
        ReLUNodeInfo: ReLUQuadPropNode,
        SigmoidNodeInfo: SigmoidQuadPropNode,
        TanhNodeInfo: TanhQuadPropNode,
        MatMulCstNodeInfo: MatMulCstQuadPropNode,
        ConcatNodeInfo: ConcatQuadPropNode,
        SliceNodeInfo: SliceQuadPropNode,
    }
)
_BATCHED_QUADPROP_NODES = types.MappingProxyType(
    {
        FlattenNodeInfo: BatchedFlattenQuadPropNode,
        GEMMNodeInfo: BatchedGEMMQuadPropNode,
        InputNodeInfo: BatchedInputQuadPropNode,
        ReLUNodeInfo: BatchedReLUQuadPropNode,
        OutputNodeInfo: BatchedOutputQuadPropNode,
        MatMulCstNodeInfo: BatchedMatMulCstQuadPropNode,
        ConcatNodeInfo: BatchedConcatQuadPropNode,
        SliceNodeInfo: BatchedSliceQuadPropNode,
    }
)
_LINPROG_NODES = types.MappingProxyType(
    {
        Conv2dNodeInfo: Conv2dLinProgNode,
        FlattenNodeInfo: FlattenLinProgNode,
        GEMMNodeInfo: GEMMLinProgNode,
        InputNodeInfo: InputLinProgNode,
        MaxPool2dNodeInfo: MaxPool2dLinProgNode,
        OutputNodeInfo: OutputLinProgNode,
        ReLUNodeInfo: ReLULinProgNode,
        AddNodeInfo: AddLinProgNode,
    }
)
_NODE_MAP = types.MappingProxyType(
    {
        LinPropArgs: _LINPROP_NODES,
        BatchedLinPropArgs: _BATCHED_LINPROP_NODES,
        AdaPropArgs: _ADAPROP_NODES,
        BatchedAdaPropArgs: _BATCHED_ADAPROP_NODES,
        QuadPropArgs: _QUADPROP_NODES,
        BatchedQuadPropArgs: _BATCHED_QUADPROP_NODES,
        InputSplitPropArgs: _LINPROP_NODES,
        BatchedInputSplitPropArgs: _BATCHED_LINPROP_NODES,
        LinProgArgs: _LINPROG_NODES,
    }
)


def build_nodes_by_infos(
    node_infos: list[BasicNodeInfo], arguments: BasicArgs
) -> list[BasicNode]:

    node_map = _NODE_MAP.get(type(arguments))
    if node_map is None:
        raise NotImplementedError(f"Unsupported arguments type: {type(arguments)}")

    Cache = _CACHE_MAP.get(type(arguments))
    if Cache is None:
        raise NotImplementedError(f"Unsupported arguments type: {type(arguments)}")
    if arguments.dtype == "float32":
        dtype = torch.float32
    elif arguments.dtype == "float64":
        dtype = torch.float64
    else:
        raise NotImplementedError(f"Unsupported dtype: {arguments.dtype}")
    device = torch.device(arguments.device)
    random_seed = arguments.random_seed
    cache: BasicCache = Cache(dtype=dtype, device=device, random_seed=random_seed)

    nodes = []
    for node_info in node_infos:
        Node = node_map.get(type(node_info))
        if Node is None:
            raise NotImplementedError(
                f"Unsupported node type: {type(node_info)} for {type(arguments)}"
            )

        node = Node(node_info, cache, arguments)
        nodes.append(node)

    return nodes
