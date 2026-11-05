__all__ = [
    "loss_minimize_min_or_group",
    "loss_minimize_max_or_group",
    "loss_maximize_min_or_group",
    "loss_maximize_max_or_group",
    "loss_max_sum_output",
    "loss_softmin",
]

import torch
from torch import Tensor


def loss_minimize_min_or_group(outputs: Tensor, or_groups: list[list[int]]) -> Tensor:
    group_outputs = [outputs[:, group].sum(dim=1) for group in or_groups]
    min_group_output = torch.stack(group_outputs, dim=1).min()

    return min_group_output


def loss_minimize_max_or_group(outputs: Tensor, or_groups: list[list[int]]) -> Tensor:
    group_outputs = [outputs[:, group].sum(dim=1) for group in or_groups]
    max_group_output = torch.stack(group_outputs, dim=1).max()

    return max_group_output


def loss_maximize_min_or_group(outputs: Tensor, or_groups: list[list[int]]) -> Tensor:
    group_outputs = [outputs[:, group].sum(dim=1) for group in or_groups]
    min_group_output = torch.stack(group_outputs, dim=1).min()

    return -min_group_output


def loss_maximize_max_or_group(outputs: Tensor, or_groups: list[list[int]]) -> Tensor:
    group_outputs = [outputs[:, group].sum(dim=1) for group in or_groups]
    max_group_output = torch.stack(group_outputs, dim=1).max()

    return -max_group_output


def loss_max_sum_output(outputs: Tensor, *args) -> Tensor:
    outputs = torch.sum(outputs, dim=1)

    return -outputs.max()


def loss_softmin(logits: Tensor, *args) -> Tensor:
    tau = 1.0
    softmin = -tau * torch.logsumexp(-logits / tau, dim=1)

    return -softmin.max()
