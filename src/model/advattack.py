__all__ = [
    "AdvAttackModel",
    "auto_pgd_linf",
]

import time
from typing import Callable

import onnxruntime as ort
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from src.args import AdvAttackArgs, extract_torch_kwargs
from src.enums import VerifStatus, PropStatus
from src.model.ort_session_infer import run_onnx_model
from src.model.utils import split_coeffs_and_const, create_or_groups
from src.utils.loss import *


@torch.no_grad()
def detect_loss_oscillation(
    loss_values: list[float],
    current_index: int,
    window_length: int = 10,
    increase_threshold: float = 0.75,
) -> bool:
    """Detect whether the loss is oscillating over a sliding window."""
    if current_index < window_length:
        return False

    increase_count = sum(
        loss_values[current_index - i] > loss_values[current_index - i - 1]
        for i in range(window_length)
    )
    return (increase_count / window_length) > increase_threshold


def init_input(
    original_input: Tensor,
    lower_limit: Tensor,
    upper_limit: Tensor,
    num_points: int,
):
    shape = original_input.shape[1:]
    dtype, device = original_input.dtype, original_input.device
    x = torch.rand((num_points, *shape), dtype=dtype, device=device)
    x.mul_(upper_limit - lower_limit).add_(lower_limit)

    x.detach_().requires_grad_(True)

    return x


def _find_adv_examples_idxs(unsafe: Tensor, y: Tensor) -> Tensor:
    adv_idxs = unsafe.nonzero(as_tuple=True)[0]
    adv_y_list, selected_idxs = [], []
    for idx in adv_idxs:
        new_adv_y = y[idx]
        if not all(not torch.equal(new_adv_y, adv_y) for adv_y in adv_y_list):
            continue

        adv_y_list.append(new_adv_y)
        selected_idxs.append(idx)
        if len(selected_idxs) >= 10:
            break

    adv_idxs = torch.stack(selected_idxs)

    return adv_idxs


def early_stop(
    y: Tensor, or_groups: list[list[int]], each_prop_status: list[list[PropStatus]]
) -> tuple[bool, Tensor | None]:
    """If any outputs of output constraints is negative, we have found an adversarial example."""
    if y.dim() == 1:
        y = y.unsqueeze(0)
    assert y.dim() == 2, f"Expected 2D tensor, got {y.shape}."
    assert len(or_groups) == len(each_prop_status)
    assert all(len(g) == len(p) for g, p in zip(or_groups, each_prop_status))

    for group_status in each_prop_status:
        for j in range(len(group_status)):
            group_status[j] = PropStatus.UNKNOWN

    violate = y >= 0.0
    for i, group in enumerate(or_groups):
        unsafe = violate[:, group].all(dim=1)
        if not unsafe.any():
            continue

        adv_idxs = _find_adv_examples_idxs(unsafe, y)

        for j in range(len(each_prop_status[i])):
            each_prop_status[i][j] = PropStatus.UNSAFE

        return True, adv_idxs
    return False, None


def _model_output(
    model, x: Tensor, output_weight: Tensor, output_bias: Tensor
) -> Tensor:
    y = model(x)
    return y @ output_weight.t() + output_bias


def auto_pgd_linf(
    onnx_model: ort.InferenceSession,
    torchonnx_model: nn.Module,
    original_input: Tensor,
    lower_limit: Tensor,
    upper_limit: Tensor,
    output_constr_coeffs: Tensor,
    output_constr_const: Tensor,
    *,
    or_groups: list[list[int]],
    each_prop_status: list[list[PropStatus]],
    loss_fn: Callable,
    num_steps: int = 100,
    num_restarts: int = 10,
    num_points: int = 10000,
    lr: float = 0.1,
    min_loss_change: float = 1e-4,
    verbose: bool = False,
) -> tuple[VerifStatus, tuple[Tensor, Tensor] | None]:
    """Perform the Auto-PGD adversarial attack under L∞ norm, using custom lower and upper limits."""
    if original_input.size(0) != 1:
        raise ValueError(f"Only support batch size 1, but got {original_input.size(0)}")

    torchonnx_model.eval()

    lower_limit = lower_limit.unsqueeze(0)
    upper_limit = upper_limit.unsqueeze(0)
    if verbose:
        print(f"Lower input limit: {lower_limit.tolist()}")
        print(f"Upper input limit: {upper_limit.tolist()}")

    loss_of_last_restart = float("inf")
    for restart in range(num_restarts):
        start_time = time.perf_counter()

        x = init_input(original_input, lower_limit, upper_limit, num_points)
        x.detach_().requires_grad_()

        optimizer = torch.optim.Adam([x], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )

        loss_of_last_step = float("inf")
        loss_value = float("inf")
        for step in range(num_steps):
            optimizer.zero_grad()

            y = torchonnx_model(x)
            z = y @ output_constr_coeffs.t() + output_constr_const

            if len(z.shape) == 1:
                z = z.unsqueeze(0)
            loss = loss_fn(z, or_groups)
            loss_value = loss.item()
            if verbose:
                print(f"  Step {step + 1}/{num_steps}: loss={loss_value:.4f}")
            if abs(loss_value - loss_of_last_step) < min_loss_change:
                break
            loss_of_last_step = loss_value

            if_early_stop, adv_idxs = early_stop(z, or_groups, each_prop_status)
            if if_early_stop:
                if verbose:
                    print("Check possible adv examples with ONNX model")

                adv_x = x[adv_idxs].cpu().detach().numpy()
                adv_y = run_onnx_model(onnx_model, adv_x)
                adv_y = torch.tensor(adv_y, dtype=x.dtype, device=x.device)
                adv_z = adv_y @ output_constr_coeffs.t() + output_constr_const

                if_early_stop, adv_idxs = early_stop(adv_z, or_groups, each_prop_status)
                if if_early_stop:
                    if verbose:
                        print(f"Confirm {adv_idxs.numel()} adv example(s) found")

                    adv_x = x[adv_idxs]
                    adv_y = adv_y[adv_idxs]
                    return VerifStatus.UNSAFE, (adv_x, adv_y)

            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                x.data = x.data.clamp(min=lower_limit, max=upper_limit)

        end_time = time.perf_counter()
        used_time = end_time - start_time
        if verbose:
            print(
                f"Restart {restart + 1}/{num_restarts} "
                f"({used_time:.2f}s, {num_points} points): "
                f"loss={loss_value:.4f}"
            )

        loss_of_last_restart = loss_value

    return VerifStatus.UNKNOWN, None


class AdvAttackModel:
    _arguments: AdvAttackArgs
    _onnx_model: ort.InferenceSession
    _torchonnx_model: Module
    _or_groups: list[list[int]] | None
    _each_prop_status: list[list[PropStatus]] | None
    _adv_example: tuple[Tensor, Tensor] | None
    _dtype: torch.dtype
    _device: torch.device

    arguments: AdvAttackArgs
    onnx_model: ort.InferenceSession
    torchonnx_model: Module
    or_groups: list[list[int]] | None
    each_prop_status: list[list[PropStatus]] | None
    adv_example: tuple[Tensor, Tensor] | None
    adv_attack_step_size: float
    adv_attack_max_iters: int
    dtype: torch.dtype
    device: torch.device

    def __init__(self, arguments: AdvAttackArgs, verbose: bool = False):
        self._arguments = arguments
        self._onnx_model = arguments.onnx_model
        self._torchonnx_model = arguments.torchonnx_model

        torch_kwargs = extract_torch_kwargs(arguments)
        self._dtype = torch_kwargs["dtype"]
        self._device = torch_kwargs["device"]

        self._or_groups = None
        self._each_prop_status = None
        self._adv_example = None
        self.verbose = verbose

    def set_output_constrs(self, output_constrs: list[Tensor]):
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
        initial_input: Tensor | None = None,
    ):
        """Find adversarial example for the given image and label."""
        assert input_constrs.size(1) == 2
        assert input_constrs.dim() == 2
        assert all(c.dim() == output_constrs[0].dim() for c in output_constrs)
        assert all(c.size(1) == output_constrs[0].size(1) for c in output_constrs)

        input_constrs = input_constrs.to(dtype=self.dtype, device=self.device)
        weight, bias = self.set_output_constrs(output_constrs)

        lower_limit, upper_limit = input_constrs[:, 0], input_constrs[:, 1]
        if initial_input is None:
            initial_input = (lower_limit + upper_limit) * 0.5
            initial_input.unsqueeze_(0)

        status, adv_examples = auto_pgd_linf(
            self.onnx_model,
            self.torchonnx_model,
            initial_input,
            lower_limit,
            upper_limit,
            weight,
            bias,
            or_groups=self.or_groups,
            each_prop_status=self.each_prop_status,
            loss_fn=loss_max_sum_output,
            num_restarts=self.arguments.adv_restarts,
            num_steps=self.arguments.adv_max_iters,
            num_points=self.arguments.adv_num_points,
            lr=self.arguments.adv_lr,
            min_loss_change=self.arguments.adv_min_loss_change,
            verbose=self.verbose,
        )

        if status == VerifStatus.UNSAFE:
            adv_x, adv_y = adv_examples
            self._adv_example = (adv_x.clone().detach_(), adv_y.clone().detach_())

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
    def arguments(self):
        return self._arguments

    @property
    def onnx_model(self):
        return self._onnx_model

    @property
    def torchonnx_model(self):
        return self._torchonnx_model

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device
