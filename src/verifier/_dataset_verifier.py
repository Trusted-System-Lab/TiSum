__all__ = ["DatasetVerifier"]

import os.path
import time

import torch
from torch import Tensor

from src import OriModelArgs
from src.args import BasicArgs
from src.enums import VerifStatus
from src.factory import ModelFactory
from src.verifier._datasets import load_torch_dataset, get_num_classes
from src.verifier._slimonnx import slim_onnx_by_path
from src.verifier._torchonnx import torch_onnx_by_path
from src.verifier._utils import run_models


def _check_arguments_list(arguments_list: list[BasicArgs]) -> None:
    argument = arguments_list[0]
    dataset = argument.dataset
    use_normal = argument.use_input_normal
    means = argument.input_means
    stds = argument.input_stds
    lower_limit = argument.input_lower_limit
    upper_limit = argument.input_upper_limit

    for argument in arguments_list[1:]:
        if argument.dataset != dataset:
            raise ValueError(
                f"All arguments must have the same dataset. "
                f"Expected {dataset}, but got {argument.dataset}."
            )
        if argument.use_input_normal != use_normal:
            raise ValueError(
                f"All arguments must have the same normalization setting. "
                f"Expected use_input_normal to be {use_normal}, "
                f"but got {argument.use_input_normal}."
            )
        if torch.all(argument.input_means != means):
            raise ValueError(
                f"All arguments must have the same input means. "
                f"Expected {means}, but got {argument.input_means}."
            )
        if torch.all(argument.input_stds != stds):
            raise ValueError(
                f"All arguments must have the same input standard deviations. "
                f"Expected {stds}, but got {argument.input_stds}."
            )
        if argument.input_lower_limit != lower_limit:
            raise ValueError(
                "All arguments must have the same input lower limit. "
                f"Expected {lower_limit}, but got {argument.input_lower_limit}."
            )
        if argument.input_upper_limit != upper_limit:
            raise ValueError(
                "All arguments must have the same input upper limit. "
                f"Expected {upper_limit}, but got {argument.input_upper_limit}."
            )


def gen_input_bounds(
    x: Tensor,
    perturb_radius: float,
    means: Tensor | None = None,
    stds: Tensor | None = None,
    lower_limit: float | None = None,
    upper_limit: float | None = None,
) -> Tensor:
    l = x.detach().clone().sub_(perturb_radius)
    u = x.detach().clone().add_(perturb_radius)

    if lower_limit is not None:
        l = l.clamp_min_(lower_limit)
    if upper_limit is not None:
        u = u.clamp_max_(upper_limit)

    if means is not None:
        means = means.to(dtype=x.dtype, device=x.device)
        l = l.sub_(means)
        u = u.sub_(means)

    if stds is not None:
        stds = stds.to(dtype=x.dtype, device=x.device)
        l = l.div_(stds)
        u = u.div_(stds)

    l = l.flatten()
    u = u.flatten()

    input_bounds = torch.stack((l, u), dim=1).detach()

    return input_bounds


def gen_output_constrs(
    target_label: int,
    num_classes: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> list[Tensor]:
    constrs = torch.zeros(
        (num_classes - 1, num_classes + 1), dtype=dtype, device=device
    )
    col_idxs = list(range(num_classes))
    col_idxs.remove(target_label)
    row_idxs = list(range(num_classes - 1))
    col_idxs = [c + 1 for c in col_idxs]
    constrs[row_idxs, col_idxs] = -1.0
    constrs[:, target_label + 1] = 1.0
    constrs = -constrs
    return [constrs[[i]] for i in range(constrs.shape[0])]


class DatasetVerifier:
    _arguments_list: list[BasicArgs]
    verbose: bool

    def __init__(
        self,
        target_net_path: str,
        arguments_list: list[BasicArgs],
        specified_samples: list[int] | None = None,
        num_samples: int | None = None,
        apply_slimonnx: bool = False,
        apply_torchonnx: bool = False,
        ignore_incorrectly_classified: bool = False,
        early_stop_when_multiple_models: bool = True,
        verbose: bool = False,
    ):
        self._slimonnx_path = target_net_path
        self._arguments_list = arguments_list
        self._specified_samples = specified_samples
        self._num_samples = num_samples
        self._ignore_incorrectly_classified = ignore_incorrectly_classified
        self._early_stop_when_multiple_models = early_stop_when_multiple_models
        self.verbose = verbose

        if not os.path.exists(target_net_path):
            raise FileNotFoundError(
                f"Target network path '{target_net_path}' does not exist."
            )
        _check_arguments_list(arguments_list)

        if specified_samples is None and num_samples is None:
            raise ValueError(
                "Either `specified_samples` or `num_samples` must be provided."
            )
        if specified_samples is not None and num_samples is not None:
            raise ValueError(
                "Only one of `specified_samples` or `num_samples` can be provided."
            )

        self._onnx_path = target_net_path

        if apply_slimonnx:
            t = time.perf_counter()
            slim_onnx_by_path(target_net_path, verbose=verbose)
            if verbose:
                print(f"Slim ONNX net in {time.perf_counter() - t:.2f} seconds.")
        self._slimonnx_path = target_net_path.replace(".onnx", f"_slimmed.onnx")

        if apply_torchonnx:
            t = time.perf_counter()
            torch_onnx_by_path(self._slimonnx_path, verbose=verbose)
            if verbose:
                print(f"Torch ONNX net in {time.perf_counter() - t:.2f} seconds.")

        self._torchonnx_path = self._slimonnx_path.replace(".onnx", ".py")
        if not os.path.exists(self._torchonnx_path):
            raise FileNotFoundError(
                f"Torch network '{self._torchonnx_path}' does not exist. "
                f"Please set `apply_torchonnx=True` to generate it first."
            )

        argument = arguments_list[0]
        datasets_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "datasets",
        )
        self._data_loader = load_torch_dataset(
            dataset=argument.dataset,
            saved_dir_path=datasets_dir,
            normalize=False,
            verbose=verbose,
        )

        self._dataset = argument.dataset
        self._perturb_radius = argument.perturb_radius
        self._input_means = argument.input_means
        self._input_stds = argument.input_stds
        self._input_lower_limit = argument.input_lower_limit
        self._input_upper_limit = argument.input_upper_limit

    def _build_models(self) -> list:

        if self._ignore_incorrectly_classified:
            ori_model_args = self.arguments_list[0]
            if not isinstance(ori_model_args, OriModelArgs):
                raise ValueError(
                    "The first argument must be of OriModelArgs to ignore incorrectly "
                    "classified samples."
                )

        for argument in self._arguments_list:
            argument.set_onnx_path(self.onnx_path)
            argument.set_slimonnx_path(self.slimonnx_path)
            argument.set_torchonnx_path(self.torchonnx_path)

        models = ModelFactory.build_models_by_args_list(
            self._arguments_list, self.verbose
        )

        self._ori_model = models[0].torchonnx_model
        models.pop(0)

        return models

    def verify(self):

        results_stats = {
            VerifStatus.SAFE: 0,
            VerifStatus.UNSAFE: 0,
            VerifStatus.UNKNOWN: 0,
        }
        sample_stats = {
            VerifStatus.SAFE: [],
            VerifStatus.UNSAFE: [],
            VerifStatus.UNKNOWN: [],
        }

        num_classes = get_num_classes(self._dataset)
        models = self._build_models()

        dtype = self._ori_model.dtype
        device = self._ori_model.device

        n_samples = 0
        max_sample_idx = (
            max(self._specified_samples)
            if self._specified_samples is not None
            else float("inf")
        )

        t = time.perf_counter()
        for idx, (x, label) in enumerate(self._data_loader):
            if idx > max_sample_idx:
                break
            if (
                self._specified_samples is not None
                and idx not in self._specified_samples
            ):
                continue

            if self.verbose:
                print(f"Sample-{idx}: Correctly label={label.item()}")

            x = x.to(dtype=dtype, device=device)
            label = label.to(dtype=torch.int64, device=device)
            output_label = self._ori_model(
                x.sub(self._input_means).div_(self._input_stds)
            ).argmax()
            if output_label != label:
                if self._ignore_incorrectly_classified:
                    if self.verbose:
                        print(
                            f"Sample-{idx} is incorrectly classified as "
                            f"{output_label}, "
                            f"skipping verification."
                        )
                    continue
                status = VerifStatus.UNSAFE
            else:
                x: Tensor
                input_constrs = gen_input_bounds(
                    x,
                    self._perturb_radius,
                    means=self._input_means,
                    stds=self._input_stds,
                    lower_limit=self._input_lower_limit,
                    upper_limit=self._input_upper_limit,
                )

                output_constrs = gen_output_constrs(
                    target_label=label.item(),
                    num_classes=num_classes,
                    dtype=dtype,
                    device=device,
                )

                status, _ = run_models(
                    models,
                    input_constrs=input_constrs,
                    output_constrs=output_constrs,
                    early_stop=self._early_stop_when_multiple_models,
                    verbose=self.verbose,
                )

            results_stats[status] += 1
            sample_stats[status].append(idx)
            n_samples += 1

            if self._num_samples is not None and n_samples >= self._num_samples:
                break

        if self.verbose:
            self.print_results(
                start_time=t,
                results_stats=results_stats,
                sample_stats=sample_stats,
                n_samples=n_samples,
            )

    def print_results(
        self,
        start_time: float,
        results_stats: dict[VerifStatus, int],
        sample_stats: dict[VerifStatus, list[int]],
        n_samples: int,
    ):
        cur_t = time.perf_counter() - start_time
        print(f"Results Summary: ")
        for k, v in results_stats.items():
            print(f"    {str(k):<8}: {v:>5}")
        print(f"Results Details: ")
        for k, v in sample_stats.items():
            print(f"    {str(k):<8}: {v}")

        print(f"Total samples    : {n_samples:>3}\n")
        print(f"Total used time  : {cur_t:>6.2f} s\n")
        if n_samples > 0:
            print(f"Average time per sample: {cur_t / n_samples:>6.2f} s\n")

    @property
    def onnx_path(self):
        return self._onnx_path

    @property
    def slimonnx_path(self) -> str:
        return self._slimonnx_path

    @property
    def torchonnx_path(self) -> str:
        return self._torchonnx_path

    @property
    def arguments_list(self) -> list[BasicArgs]:
        return self._arguments_list
