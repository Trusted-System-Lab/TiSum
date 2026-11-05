__all__ = ["ModelFactory"]

import onnxruntime as ort
import torch

from src.args import *
from src.batched_model import *
from src.factory._build_nodes import build_nodes_by_infos
from src.factory._parse_nodes import parse_node_infos
from src.model import *

_MODEL_MAP = {
    OriModelArgs: OriModel,
    AdvAttackArgs: AdvAttackModel,
    LinPropArgs: LinPropModel,
    BatchedLinPropArgs: BatchedLinPropModel,
    AdaPropArgs: AdaPropModel,
    BatchedAdaPropArgs: BatchedAdaPropModel,
    QuadPropArgs: QuadPropModel,
    BatchedQuadPropArgs: BatchedQuadPropModel,
    LinProgArgs: LinProgModel,
    InputSplitPropArgs: InputSplitPropModel,
    BatchedInputSplitPropArgs: BatchedInputSplitPropModel,
}


class ModelFactory:
    _onnx_model: ort.InferenceSession | None
    _torchonnx_model: torch.nn.Module | None

    def __init__(self, verbose: bool = False):
        self._onnx_model: ort.InferenceSession | None = None
        self._torchonnx_model: torch.nn.Module | None = None
        self.verbose = verbose

    def build(self, argument: BasicArgs) -> BasicModel | OriModel | AdvAttackModel:
        if type(argument) is OriModelArgs:
            argument: OriModelArgs
            ori_model = OriModel(argument, self.verbose)
            self._torchonnx_model = ori_model.torchonnx_model
            self._onnx_model = ori_model.onnx_model
            argument.set_torchonnx_model(self._torchonnx_model)
            argument.set_onnx_model(self._onnx_model)

            return ori_model

        if type(argument) is AdvAttackArgs:
            argument: AdvAttackArgs
            if self._torchonnx_model is None:
                raise ValueError("OriModel must be set before AdvAttackModel.")
            if self._onnx_model is None:
                raise ValueError("ONNX model must be set before AdvAttackModel.")
            argument.set_torchonnx_model(self._torchonnx_model)
            argument.set_onnx_model(self._onnx_model)

            return AdvAttackModel(argument, self.verbose)

        Model = _MODEL_MAP.get(type(argument))
        if Model is None:
            raise ValueError(f"Model not found for argument type: {type(argument)}")

        if isinstance(argument, BaBPropArgs):
            if self._torchonnx_model is None:
                raise ValueError("OriModel must be set before BaBPropModel.")
            argument.set_torchonnx_model(self._torchonnx_model)
            argument.set_onnx_model(self._onnx_model)

        node_infos = parse_node_infos(argument)
        nodes = build_nodes_by_infos(node_infos, argument)

        return Model(nodes, argument, self.verbose)

    @staticmethod
    def build_models_by_args_list(
        arguments: list[BasicArgs], verbose: bool = False
    ) -> list[BasicModel | OriModel | AdvAttackModel]:
        model_factory = ModelFactory(verbose)
        models = [model_factory.build(argument) for argument in arguments]
        return models

    @property
    def onnx_model(self):
        return self._onnx_model

    @onnx_model.setter
    def onnx_model(self, value: ort.InferenceSession):
        self._onnx_model = value

    @property
    def torchonnx_model(self):
        return self._torchonnx_model

    @torchonnx_model.setter
    def torchonnx_model(self, value: torch.nn.Module):
        self._torchonnx_model = value
