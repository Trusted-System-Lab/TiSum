__all__ = ["OriModel"]

import os

import onnxruntime as ort
import torch
from torch import Tensor

from src import extract_torch_kwargs
from src.args import OriModelArgs
from src.enums import PropStatus, VerifStatus
from src.model.ort_session_infer import run_onnx_model
from src.model.utils import (
    split_coeffs_and_const,
    create_or_groups,
    analyze_each_prop_status_by_upper_bound,
    load_module,
    analyze_status,
)


class OriModel:
    _arguments: OriModelArgs
    _onnx_path: str
    _torchonnx_path: str
    _torchonnx_param_path: str
    _torchonnx_class_name: str
    _onnx_model: ort.InferenceSession
    _torchonnx_model: torch.nn.Module
    _or_groups: list[list[int]] | None
    _each_prop_status: list[list[PropStatus]] | None
    _adv_example: tuple[Tensor, Tensor] | None
    _dtype: torch.dtype
    _device: torch.device

    or_groups: list[list[int]] | None
    each_prop_status: list[list[PropStatus]] | None
    adv_example: tuple[Tensor, Tensor] | None

    def __init__(self, arguments: OriModelArgs, verbose: bool = False):
        if arguments.onnx_path is None:
            raise ValueError("onnx_path is not provided.")
        if arguments.torchonnx_path is None:
            raise ValueError("torchonnx_path is not provided.")
        if arguments.torchonnx_param_path is None:
            raise ValueError("torchonnx_param_path is not provided.")
        if not os.path.exists(arguments.onnx_path):
            raise FileNotFoundError(f"{arguments.onnx_path} does not exist.")
        if not os.path.exists(arguments.torchonnx_path):
            raise FileNotFoundError(f"{arguments.torchonnx_path} does not exist.")
        if not os.path.exists(arguments.torchonnx_param_path):
            raise FileNotFoundError(f"{arguments.torchonnx_param_path} does not exist.")

        self._arguments = arguments
        self._onnx_path = arguments.onnx_path
        self._torchonnx_path = arguments.torchonnx_path
        self._torchonnx_param_path = arguments.torchonnx_param_path
        self._torchonnx_class_name = arguments.torchonnx_class_name

        torch_kwargs = extract_torch_kwargs(arguments)
        self._dtype = torch_kwargs["dtype"]
        self._device = torch_kwargs["device"]

        self.load_onnx_model()
        self.load_torchonnx_model()

        self._or_groups = None
        self._each_prop_status = None
        self._adv_example = None
        self.verbose = verbose

    def load_onnx_model(self):
        self._onnx_model = ort.InferenceSession(self.onnx_path)

    def load_torchonnx_model(self):
        module = load_module(self.torchonnx_path, self.torchonnx_class_name)
        TempModule = getattr(module, self.torchonnx_class_name)

        self._torchonnx_model = TempModule(
            dtype=self.dtype,
            device=self.device,
            params_path=self.torchonnx_param_path,
        )
        self.torchonnx_model.eval()

    def check_model_consistency(self, x: Tensor, y: Tensor):
        y_torch = self.torchonnx_model(x)
        if not torch.allclose(y, y_torch, atol=1e-5):
            raise RuntimeError(
                "Mismatch between ONNX model and TorchONNX model outputs. "
                "Please check the exported models for consistency."
            )

    def set_output_constrs(self, output_constrs: list[Tensor]) -> tuple[Tensor, Tensor]:
        assert all(c.dim() == output_constrs[0].dim() for c in output_constrs)

        weight, bias = split_coeffs_and_const(output_constrs)
        weight = weight.to(dtype=self.dtype, device=self.device)
        bias = bias.to(dtype=self.dtype, device=self.device)

        or_groups = create_or_groups(output_constrs)
        self._or_groups = or_groups
        self.each_prop_status = [[PropStatus.UNKNOWN] * len(g) for g in or_groups]
        self._adv_example = None

        return weight, bias

    def run(
        self,
        input_constrs: Tensor,
        output_constrs: list[Tensor],
        *,
        initial_input: Tensor = None,
    ):
        assert input_constrs.size(1) == 2
        assert input_constrs.dim() == 2
        assert all(c.dim() == output_constrs[0].dim() for c in output_constrs)
        assert all(c.size(1) == output_constrs[0].size(1) for c in output_constrs)

        input_constrs = input_constrs.to(dtype=self.dtype, device=self.device)
        weight, bias = self.set_output_constrs(output_constrs)

        if initial_input is None:
            initial_input = (input_constrs[:, 0] + input_constrs[:, 1]) * 0.5

        x = initial_input
        with torch.no_grad():
            y = run_onnx_model(self.onnx_model, x.cpu().numpy())
            y = torch.tensor(y, dtype=self._dtype, device=self._device)
            self.check_model_consistency(x, y)
            y = y.reshape(1, -1)
            z = y @ weight.t() + bias

        z = (-z).squeeze()
        if z.dim() == 0:
            z = z.unsqueeze(0)
        each_prop_status = analyze_each_prop_status_by_upper_bound(z, self.or_groups)
        status = analyze_status(each_prop_status)
        if status == VerifStatus.UNSAFE:
            self._adv_example = (x.clone().detach_(), y.clone().detach_())

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
    def arguments(self) -> OriModelArgs:
        return self._arguments

    @property
    def onnx_path(self) -> str:
        return self._onnx_path

    @property
    def onnx_model(self) -> ort.InferenceSession | None:
        return self._onnx_model

    @property
    def torchonnx_path(self) -> str:
        return self._torchonnx_path

    @property
    def torchonnx_param_path(self) -> str:
        return self._torchonnx_param_path

    @property
    def torchonnx_class_name(self) -> str:
        return self._torchonnx_class_name

    @property
    def torchonnx_model(self):
        return self._torchonnx_model

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device
