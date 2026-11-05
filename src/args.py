"""This module contains the arguments for the experiment."""

__all__ = [
    "BasicArgs",
    "OriModelArgs",
    "AdvAttackArgs",
    "LinPropArgs",
    "AdaPropArgs",
    "QuadPropArgs",
    "LinProgArgs",
    "BaBPropArgs",
    "InputSplitPropArgs",
    "BatchedInputSplitPropArgs",
    "BatchedLinPropArgs",
    "BatchedAdaPropArgs",
    "BatchedQuadPropArgs",
    "extract_torch_kwargs",
]


import os
import random
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import onnxruntime as ort
import torch
from torch import Tensor

from propdag import TArgument, PropMode
from src.enums import *
from src.substorch import RelaxMode, ConstrTemplate
from src.torchonnx import gen_module_class_name


def _set_cal_settings(random_seed: int, device: str, dtype: str):
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=200, profile="full")
    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    torch.autograd.set_detect_anomaly(True)

    if dtype == "float64":
        torch.set_default_dtype(torch.float64)
    elif dtype == "float32":
        torch.set_default_dtype(torch.float32)
    else:
        raise ValueError(f"Invalid dtype: {dtype}.")

    if device != "cpu":
        torch.cuda.manual_seed(random_seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = dtype == "float32"
        torch.backends.cudnn.allow_tf32 = dtype == "float32"


def _get_means_stds_limits(
    dataset: str,
    net_file_name: str,
    means: Tensor | None = None,
    stds: Tensor | None = None,
    lower_limit: float | None = None,
    upper_limit: float | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> tuple[bool, Tensor, Tensor, float, float]:
    """Set the means and standard deviations for normalization."""
    if lower_limit is None and dataset in {"mnist", "cifar10"}:
        lower_limit = 0.0
    if upper_limit is None and dataset in {"mnist", "cifar10"}:
        upper_limit = 1.0

    if means is None:
        if dataset == "mnist":
            means = torch.tensor([0.0])
        elif dataset == "cifar10":
            means = torch.tensor(
                [
                    [[0.49140000343322754]],
                    [[0.4821999967098236]],
                    [[0.4465000033378601]],
                ]
            )

    if stds is None:
        if dataset == "mnist":
            stds = torch.tensor([1.0])
        elif dataset == "cifar10":
            stds = torch.tensor(
                [
                    [[0.20229999721050262]],
                    [[0.19939999282360077]],
                    [[0.20100000500679016]],
                ]
            )

    if net_file_name in {
        "mnist_sigmoid_6_500",
        "mnist_tanh_6_500",
        "mnist_sigmoid_convmed",
        "mnist_tanh_convmed",
        "mnist_convSmallRELU__Point",
    }:
        print("Using specific means and stds for the network.")
        means = torch.tensor([0.1307])
        stds = torch.tensor([0.3081])

    normalize = True
    if net_file_name in {
        "mnist_sigmoid_6_500",
        "mnist_tanh_6_500",
        "mnist_sigmoid_convmed",
        "mnist_tanh_convmed",
        "mnist_convSmallRELU__Point",
        "cifar_sigmoid_6_500",
        "cifar_tanh_6_500",
        "cifar_sigmoid_convmed",
        "cifar_tanh_convmed",
    }:
        print("Not using normalization for the network.")
        normalize = False

    means = means.to(dtype=dtype, device=device)
    stds = stds.to(dtype=dtype, device=device)
    return normalize, means, stds, lower_limit, upper_limit


@dataclass(frozen=True, slots=True)
class BasicArgs(TArgument):
    onnx_path: str | None = None
    onnx_model: ort.InferenceSession | None = None
    slimonnx_path: str | None = None
    use_vnnlib: bool = False
    use_dataset: bool = False
    dataset: str | None = None
    perturb_radius: float | None = None
    perturb_norm: float = float("inf")
    first_sample_index: int = 0
    num_samples: int = 1
    check_ignored_samples: bool = True
    use_input_normal: bool = True
    input_means: torch.Tensor | None = None
    input_stds: torch.Tensor | None = None
    input_lower_limit: float | None = None
    input_upper_limit: float | None = None
    random_seed: int = 0
    dtype: str = "float32"
    device: str = "cpu"
    nodes_sort_strategy: Literal["bfs", "dfs"] = "bfs"
    torchonnx_path: str | None = None
    torchonnx_model: torch.nn.Module | None = None
    torchonnx_class_name: str | None = None
    torchonnx_param_path: str | None = None
    store_all_bnds: bool = False
    store_output_final_symbnds: bool = False
    find_adv_example: bool = False

    def __post_init__(self):
        if self.use_dataset:
            if self.dataset is None:
                raise ValueError("dataset should be given.")
            if self.perturb_radius is None:
                raise ValueError("perturb_radius should be given.")
            if self.perturb_radius < 0:
                raise ValueError(f"Invalid {self.perturb_radius}.")
            self.set_normalization()

        if not self.use_vnnlib and not self.use_dataset:
            raise ValueError("Either use_vnnlib or use_dataset should be True.")

        if self.nodes_sort_strategy not in ["bfs", "dfs"]:
            raise ValueError(
                f"Invalid nodes_sort_strategy {self.nodes_sort_strategy}. "
                f"It should be either 'bfs' or 'dfs'."
            )

        _set_cal_settings(self.random_seed, self.device, self.dtype)

    def set_normalization(self):
        if not self.use_dataset:
            return

        if self.slimonnx_path is None:
            net_file_name = ""
        else:
            net_file_name = os.path.basename(self.slimonnx_path).split(".")[0]
        use_normal, means, stds, lower_limit, upper_limit = _get_means_stds_limits(
            self.dataset,
            net_file_name,
            self.input_means,
            self.input_stds,
            self.input_lower_limit,
            self.input_upper_limit,
            torch.float64 if self.dtype == "float64" else torch.float32,
            torch.device(self.device),
        )
        object.__setattr__(self, "use_input_normal", use_normal)
        object.__setattr__(self, "input_means", means)
        object.__setattr__(self, "input_stds", stds)
        object.__setattr__(self, "input_lower_limit", lower_limit)
        object.__setattr__(self, "input_upper_limit", upper_limit)

    def set_onnx_path(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Network file {path} does not exist.")
        object.__setattr__(self, "onnx_path", path)

    def set_onnx_model(self, model: ort.InferenceSession):
        if not isinstance(model, ort.InferenceSession):
            raise TypeError("model should be an instance of ort.InferenceSession.")
        object.__setattr__(self, "onnx_model", model)

    def set_slimonnx_path(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Network file {path} does not exist.")
        object.__setattr__(self, "slimonnx_path", path)
        self.set_normalization()

    def set_torchonnx_path(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Module file {path} does not exist.")
        object.__setattr__(self, "torchonnx_path", path)
        class_name = gen_module_class_name(path)
        object.__setattr__(self, "torchonnx_class_name", class_name)
        param_path = path.replace(".py", ".pth")
        if not os.path.exists(param_path):
            raise FileNotFoundError(f"Parameter file {param_path} does not exist.")
        object.__setattr__(self, "torchonnx_param_path", param_path)

    def set_torchonnx_model(self, model: torch.nn.Module):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"model should be an instance of torch.nn.Module but got {type(model)}."
            )
        object.__setattr__(self, "torchonnx_model", model)


@dataclass(frozen=True, slots=True)
class OriModelArgs(BasicArgs):
    def __post_init__(self):
        BasicArgs.__post_init__(self)


@dataclass(frozen=True, slots=True)
class AdvAttackArgs(OriModelArgs):
    adv_lr: float = 0.1
    adv_min_loss_change: float = 1e-4
    adv_restarts: int = 5
    adv_num_points: int = 100000
    adv_max_iters: int = 10

    def __post_init__(self):
        OriModelArgs.__post_init__(self)
        if self.adv_lr <= 0:
            raise ValueError(f"Invalid {self.adv_lr}.")
        if self.adv_min_loss_change < 0:
            raise ValueError(f"Invalid {self.adv_min_loss_change}.")
        if self.adv_max_iters <= 0:
            raise ValueError(f"Invalid {self.adv_max_iters}.")


@dataclass(frozen=True, slots=True)
class LinPropArgs(BasicArgs):
    bound_prop_mode: BoundPropMode = BoundPropMode.INEQ
    act_relax_mode: RelaxMode = RelaxMode.SINGLE_LINEAR
    act_relax_min_range: float = 0.1
    sigmoid_relax_limit: float = 5.0
    tanh_relax_limit: float = 5.0
    refine_by_scalarize_each_layer: bool = False
    refine_by_interval_arithmetic: bool = True

    def __post_init__(self):
        BasicArgs.__post_init__(self)
        if self.act_relax_min_range < 0:
            raise ValueError(f"Invalid {self.act_relax_min_range}.")
        if self.sigmoid_relax_limit < 0:
            raise ValueError(f"Invalid {self.sigmoid_relax_limit}.")
        if self.tanh_relax_limit < 0:
            raise ValueError(f"Invalid {self.tanh_relax_limit}.")


@dataclass(frozen=True, slots=True)
class BatchedLinPropArgs(LinPropArgs):
    max_batched_size: int = 16

    def __post_init__(self):
        LinPropArgs.__post_init__(self)
        if self.max_batched_size <= 0:
            raise ValueError(f"Invalid {self.max_batched_size}.")


@dataclass(frozen=True, slots=True)
class AdaPropArgs(LinPropArgs):
    ada_optimizer: str = "adam"
    ada_lr: float = 0.1
    ada_weight_decay: float = 0.0
    ada_max_iters: int = 5
    ada_min_loss_change: float = 0.001
    ada_max_opt_layers: int = -1
    ada_only_last_layer: bool = True

    def __post_init__(self):
        LinPropArgs.__post_init__(self)
        if self.ada_lr <= 0:
            raise ValueError(f"Invalid {self.ada_lr}.")
        if self.ada_weight_decay < 0:
            raise ValueError(f"Invalid {self.ada_weight_decay}.")
        if self.ada_max_iters < 0:
            raise ValueError(f"Invalid {self.ada_max_iters}.")
        if self.ada_min_loss_change < 0:
            raise ValueError(f"Invalid {self.ada_min_loss_change}.")
        if self.ada_max_opt_layers < -1:
            raise ValueError(f"Invalid {self.ada_max_opt_layers}.")


@dataclass(frozen=True, slots=True)
class BatchedAdaPropArgs(AdaPropArgs):
    max_batched_size: int = 16

    def __post_init__(self):
        AdaPropArgs.__post_init__(self)
        if self.max_batched_size <= 0:
            raise ValueError(f"Invalid {self.max_batched_size}.")


@dataclass(frozen=True, slots=True)
class BaBPropArgs(AdaPropArgs):
    bab_timeout: float = 999
    bab_max_cached_branches: int = -1
    bab_max_depth: int = 30
    bab_max_paral_branches: int = 4
    bab_min_bound_change: float = 1e-4
    bab_min_range: float = 1e-6
    bab_branch_queue_priority_mode: BranchPriorMode = BranchPriorMode.MIN_DISTANCE_ZERO
    bab_max_size_for_quadprop: int = 500

    def __post_init__(self):
        AdaPropArgs.__post_init__(self)
        if self.bab_timeout <= 0:
            raise ValueError(f"Invalid {self.bab_timeout}.")
        if self.bab_max_cached_branches < -1:
            raise ValueError(f"Invalid {self.bab_max_cached_branches}.")
        if self.bab_min_bound_change < 0:
            raise ValueError(f"Invalid {self.bab_min_bound_change}.")
        if self.bab_min_range < 0:
            raise ValueError(f"Invalid {self.bab_min_range}.")
        if self.bab_max_paral_branches <= -1:
            raise ValueError(f"Invalid {self.bab_max_paral_branches}.")
        if self.bab_max_size_for_quadprop < -1:
            raise ValueError(f"Invalid {self.bab_max_size_for_quadprop}.")


@dataclass(frozen=True, slots=True)
class InputSplitPropArgs(BaBPropArgs):
    bab_split_mode: SplitMode = SplitMode.BISECTION
    bab_use_parent_branch_caches: bool = True
    bab_max_depth_for_only_split: int = 2
    bab_max_range_for_only_split: float = 100
    bab_min_range_for_only_split: float = 0.1
    bab_max_cached_branches_for_only_split: int = 1000

    bab_use_adv_attack_before_prop: bool = True
    bab_thr_to_try_adv_attack: float = 0.01
    bab_adv_lr: float = 0.01
    bab_adv_min_loss_change: float = 1e-3
    bab_adv_restarts: int = 3
    bab_adv_num_points: int = 100
    bab_adv_max_iters: int = 3

    bab_use_direct_symbols: bool = True
    bab_use_linprop: bool = True
    bab_use_quadprop: bool = False
    bab_use_adaprop: bool = False
    bab_ada_only_last_layer: bool = True

    bab_use_linprog_when_small_range: bool = True
    bab_use_linprog_min_depth: int = 10
    bab_use_linprog_max_depth: int = 12
    bab_use_mip_min_depth: int = 10
    bab_use_mip_max_depth: int = 15
    bab_lp_min_range_mip_relu: float = 0.0
    bab_use_linprog_max_num_mip_relu: int = float("inf")
    bab_use_linprog_with_mip: bool = True
    bab_use_linprog_timelimit: float = 5.0
    bab_thr_to_try_linprog: float = 0.1
    bab_thr_to_try_mip: float = 0.1
    num_samples_for_grad_estimation: int = 1000

    def __post_init__(self):
        BaBPropArgs.__post_init__(self)
        if self.bab_max_depth_for_only_split < -1:
            raise ValueError(f"Invalid {self.bab_max_depth_for_only_split}.")
        if self.bab_max_range_for_only_split < 0:
            raise ValueError(f"Invalid {self.bab_max_range_for_only_split}.")
        if self.bab_min_range_for_only_split < 0:
            raise ValueError(f"Invalid {self.bab_min_range_for_only_split}.")
        if self.bab_max_cached_branches_for_only_split < -1:
            raise ValueError(f"Invalid {self.bab_max_cached_branches_for_only_split}.")
        if self.bab_thr_to_try_adv_attack < 0:
            raise ValueError(f"Invalid {self.bab_thr_to_try_adv_attack}.")
        if self.bab_adv_lr <= 0:
            raise ValueError(f"Invalid {self.bab_adv_lr}.")
        if self.bab_adv_min_loss_change < 0:
            raise ValueError(f"Invalid {self.bab_adv_min_loss_change}.")
        if self.bab_adv_restarts < 0:
            raise ValueError(f"Invalid {self.bab_adv_restarts}.")
        if self.bab_adv_num_points < 0:
            raise ValueError(f"Invalid {self.bab_adv_num_points}.")
        if self.bab_adv_max_iters < 0:
            raise ValueError(f"Invalid {self.bab_adv_max_iters}.")
        if self.bab_use_linprog_min_depth < 0:
            raise ValueError(f"Invalid {self.bab_use_linprog_min_depth}.")
        if self.bab_use_linprog_max_depth < 0:
            raise ValueError(f"Invalid {self.bab_use_linprog_max_depth}.")
        if self.bab_use_linprog_max_depth < self.bab_use_linprog_min_depth:
            raise ValueError(
                f"Invalid {self.bab_use_linprog_max_depth} less than "
                f"{self.bab_use_linprog_min_depth}."
            )
        if self.bab_use_mip_min_depth < 0:
            raise ValueError(f"Invalid {self.bab_use_mip_min_depth}.")
        if self.bab_use_mip_max_depth < 0:
            raise ValueError(f"Invalid {self.bab_use_mip_max_depth}.")
        if self.bab_use_mip_max_depth < self.bab_use_mip_min_depth:
            raise ValueError(
                f"Invalid {self.bab_use_mip_max_depth} less than "
                f"{self.bab_use_mip_min_depth}."
            )
        if self.bab_use_linprog_timelimit < 0:
            raise ValueError(f"Invalid {self.bab_use_linprog_timelimit}.")
        if self.bab_thr_to_try_linprog < 0:
            raise ValueError(f"Invalid {self.bab_thr_to_try_linprog}.")
        if self.bab_thr_to_try_mip < 0:
            raise ValueError(f"Invalid {self.bab_thr_to_try_mip}.")


@dataclass(frozen=True, slots=True)
class BatchedInputSplitPropArgs(InputSplitPropArgs):
    bab_max_paral_linprog_models: int = 1000

    def __post_init__(self):
        InputSplitPropArgs.__post_init__(self)
        if self.bab_max_paral_linprog_models < 1:
            raise ValueError(f"Invalid {self.bab_max_paral_linprog_models}.")


@dataclass(frozen=True, slots=True)
class QuadPropArgs(LinPropArgs):
    quad_rlx_min_range: float = 0.01
    quad_max_batches: int = -1
    quad_max_paral_batches: int = 32

    def __post_init__(self):
        LinPropArgs.__post_init__(self)
        if self.quad_rlx_min_range < 0:
            raise ValueError(f"Invalid {self.quad_rlx_min_range}.")
        if self.quad_max_batches < -1:
            raise ValueError(f"Invalid {self.quad_max_batches}.")
        if self.quad_max_paral_batches < -1:
            raise ValueError(f"Invalid {self.quad_max_paral_batches}.")


@dataclass(frozen=True, slots=True)
class BatchedQuadPropArgs(QuadPropArgs):
    max_batched_size: int = 16

    def __post_init__(self):
        QuadPropArgs.__post_init__(self)
        if self.max_batched_size <= 0:
            raise ValueError(f"Invalid {self.max_batched_size}.")


@dataclass(frozen=True, slots=True)
class LinProgArgs(LinPropArgs):
    lp_solver: str = "GRB"
    use_lin_prog_terminate_if_fail: bool = True
    use_mip: bool = False
    grb_output_flag: int = 0
    lp_timelimit: float = 1
    grb_numeric_focus: int = 0
    grb_optimality_tol: float = 1e-9
    grb_feasibility_tol: float = 1e-4
    grb_markowitz_tol: float = 0.99
    grb_method: int = 0
    lp_lower_cutoff: float = 1e-6
    lp_upper_cutoff: float = 0.0
    lp_max_num_mip_relu: int = float("inf")
    lp_min_range_mip_relu: float = 0.0

    def __post_init__(self):
        LinPropArgs.__post_init__(self)
        object.__setattr__(self, "prop_mode", PropMode.BACKWARD)

    def set_use_mip(self, value: bool):
        object.__setattr__(self, "use_mip", value)


_dtype = None
_device = None


def extract_torch_kwargs(args: BasicArgs) -> dict[str, Any]:
    global _dtype, _device
    if _dtype is None and _device is None:
        if args.dtype == "float32":
            _dtype = torch.float32
        elif args.dtype == "float64":
            _dtype = torch.float64
        else:
            raise ValueError(f"Invalid dtype {args.dtype}.")
        if args.device == "cpu":
            _device = torch.device("cpu")
        elif "cuda" in args.device:
            _device = torch.device(args.device)
        elif args.device == "mps":
            _device = torch.device("mps")
        else:
            raise ValueError(f"Invalid device {args.device}.")

    return {"dtype": _dtype, "device": _device}


if __name__ == "__main__":
    print("Testing args.py")
    BasicArgs()
    AdvAttackArgs()
    LinPropArgs()
    AdaPropArgs()
    BaBPropArgs()
    InputSplitPropArgs()
    QuadPropArgs()
    LinProgArgs()
