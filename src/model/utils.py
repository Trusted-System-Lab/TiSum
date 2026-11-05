__all__ = [
    "set_pre_next_nodes",
    "split_coeffs_and_const",
    "create_or_groups",
    "create_batched_or_groups",
    "analyze_each_prop_status",
    "analyze_batched_each_prop_status",
    "analyze_each_prop_status_by_lower_bound",
    "analyze_each_prop_status_by_upper_bound",
    "find_adv_example_by_final_symbnds",
    "analyze_status",
    "load_module",
]

import importlib
import os
import sys
from itertools import count, islice

import onnxruntime as ort
import torch
from torch import Tensor

from src.enums import PropStatus, VerifStatus
from src.model.ort_session_infer import run_onnx_model
from src.nodes import BasicNode
from src.substorch import find_extreme_point_of_linear_symbnds


def set_pre_next_nodes(nodes: list[BasicNode]):
    nodes_dict = {name: node for node in nodes for name in node.output_names}
    assert len(nodes_dict) == sum(
        len(node.output_names) for node in nodes
    ), "Duplicated output names found among nodes."

    for node in nodes:
        node.pre_nodes = [
            nodes_dict[name] for name in node.input_names if name in nodes_dict
        ]
        node.next_nodes = [
            next_node
            for next_node in nodes
            if any(name in next_node.input_names for name in node.output_names)
        ]


def split_coeffs_and_const(constrs: list[Tensor]) -> tuple[Tensor, Tensor]:
    constrs = torch.cat(constrs, dim=0)
    coeffs = constrs[:, 1:]
    const = constrs[:, 0]

    return coeffs, const


def create_or_groups(output_constrs: list[Tensor]) -> list[list[int]]:
    counter = count()
    or_groups = [list(islice(counter, g.size(0))) for g in output_constrs]

    return or_groups


def create_batched_or_groups(
    output_constrs: list[Tensor], batch_size: int
) -> list[list[int]]:
    or_groups = create_or_groups(output_constrs) * batch_size
    new_or_groups, i = [], 0
    for or_group in or_groups:
        new_or_groups.append(list(range(i, i + len(or_group))))
        i += len(or_group)

    return new_or_groups


def analyze_each_prop_status(
    output_bounds: tuple[Tensor, ...], or_groups: list[list[int]]
) -> list[list[PropStatus]]:
    assert len(output_bounds) == 2
    assert output_bounds[0].size(0) == sum(len(g) for g in or_groups)
    assert output_bounds[1].size(0) == sum(len(g) for g in or_groups)

    each_prop_status = []
    for group in or_groups:
        group_status = [
            (
                PropStatus.SAFE
                if output_bounds[0][i].item() > 0.0
                else (
                    PropStatus.UNSAFE
                    if output_bounds[1][i].item() <= 0.0
                    else PropStatus.UNKNOWN
                )
            )
            for i in group
        ]
        each_prop_status.append(group_status)

    return each_prop_status


def analyze_batched_each_prop_status(
    output_bounds: tuple[Tensor, ...], or_groups: list[list[int]]
) -> list[list[PropStatus]]:
    lower_output_bound = output_bounds[0]
    upper_output_bound = output_bounds[1]
    assert lower_output_bound.dim() == 2
    assert upper_output_bound.dim() == 2

    lower_output_bound = lower_output_bound.flatten()
    upper_output_bound = upper_output_bound.flatten()
    output_bounds = (lower_output_bound, upper_output_bound)

    return analyze_each_prop_status(output_bounds, or_groups)


def analyze_each_prop_status_by_lower_bound(
    lower_bound: Tensor, or_groups: list[list[int]]
) -> list[list[PropStatus]]:
    assert lower_bound.size(0) == sum(len(g) for g in or_groups)

    each_prop_status = []
    for group in or_groups:
        group_status = [
            (PropStatus.SAFE if lower_bound[i].item() > 0.0 else PropStatus.UNKNOWN)
            for i in group
        ]
        each_prop_status.append(group_status)

    return each_prop_status


def analyze_each_prop_status_by_upper_bound(
    upper_bound: Tensor, or_groups: list[list[int]]
) -> list[list[PropStatus]]:
    assert upper_bound.size(0) == sum(len(g) for g in or_groups)

    each_prop_status = []
    for group in or_groups:
        group_status = [
            (
                PropStatus.UNSAFE
                if upper_bound[i].item() <= 0.0
                else PropStatus.UNKNOWN
            )
            for i in group
        ]
        each_prop_status.append(group_status)

    return each_prop_status


def analyze_status(each_prop_status: list[list[PropStatus]]) -> VerifStatus:
    if all(any(s == PropStatus.SAFE for s in group) for group in each_prop_status):
        return VerifStatus.SAFE
    if any(all(s == PropStatus.UNSAFE for s in group) for group in each_prop_status):
        return VerifStatus.UNSAFE
    return VerifStatus.UNKNOWN


def find_adv_example_by_final_symbnds(
    symbnd: Tensor,
    input_bound: tuple[Tensor, Tensor],
    or_groups: list[list[int]],
    model: ort.InferenceSession,
) -> tuple[Tensor, Tensor] | None:
    adv_x = find_extreme_point_of_linear_symbnds(symbnd, *input_bound, is_lower=False)

    adv_x = adv_x.cpu().to(torch.float32).numpy()
    adv_y = run_onnx_model(model, adv_x)
    dtype, device = symbnd.dtype, symbnd.device
    adv_x = torch.tensor(adv_x, dtype=dtype, device=device)
    adv_y = torch.tensor(adv_y, dtype=dtype, device=device)

    for start, end in ((g[0], g[-1] + 1) for g in or_groups):
        if torch.all(adv_y[start:end] <= 0.0):
            return adv_x, adv_y
    return None


def load_module(path: str, class_name: str = "custom_module"):
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location(class_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[class_name] = module
    spec.loader.exec_module(module)

    return module
