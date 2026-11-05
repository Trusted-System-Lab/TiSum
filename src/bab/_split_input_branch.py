__all__ = ["SplitInputBranch"]

from copy import deepcopy

import torch.nn as nn
from torch import Tensor

from src.bab._branch import Branch
from src.bab._split_input import *
from src.cache import BaBPropCache
from src.enums import *
from src.model.utils import analyze_status
from src.nodes import BasicNode


class SplitInputBranch(Branch):
    _name: str
    _priority: float
    _or_groups: list[list[int]]
    _nodes: list[BasicNode]
    _cache: BaBPropCache
    _split_inputs: SplitInputDict
    name: str
    priority: float
    or_groups: list[list[int]]
    nodes: list[BasicNode]
    cache: BaBPropCache
    split_inputs: SplitInputDict
    verbose: bool

    _each_prop_status: list[list[PropStatus]]
    each_prop_status: list[list[PropStatus]]

    def __init__(
        self,
        name: str,
        priority: float,
        or_groups: list[list[int]],
        nodes: list[BasicNode],
        cache: BaBPropCache,
        split_inputs: SplitInputDict,
        *,
        verbose: bool = False,
    ):
        Branch.__init__(self, name, priority, or_groups, nodes, cache, verbose=verbose)
        self._split_inputs = split_inputs

    def _build_children_split_inputs(
        self,
        split_mode: SplitMode,
        model: nn.Module,
        num_samples_for_grad_estimation: int,
    ):

        input_bound: tuple[Tensor, Tensor] = self.cache.bnds[self.nodes[0].name]
        if self.cache.output_final_symbnds is not None:
            input_bound = clamp_intput_bounds_by_lower_output_final_symbnds(
                input_bound, self.cache.output_final_symbnds, self.or_groups,
                verbose=self.verbose,
            )
            self.cache.bnds[self.nodes[0].name] = input_bound

        if split_mode == SplitMode.BISECTION:
            return split_input_by_bisection(
                input_bound, self.split_inputs, self.verbose
            )

        if split_mode == SplitMode.EFFECT:
            output_final_symbnds = self.cache.output_final_symbnds
            assert output_final_symbnds is not None
            return split_input_by_effect(
                input_bound,
                self.split_inputs,
                output_final_symbnds,
                self.verbose,
            )

        if split_mode == SplitMode.GRAD:
            assert model is not None
            output_constrs = self.output_constrs
            assert output_constrs is not None
            return split_input_by_grad(
                input_bound,
                self.split_inputs,
                model,
                output_constrs,
                num_samples=num_samples_for_grad_estimation,
                verbose=self.verbose,
            )

        raise ValueError(f"Unsupported split mode {split_mode}.")

    def _cal_children_priorities(self, priority_mode: BranchPriorMode) -> list[float]:
        output_bound: tuple[Tensor, Tensor] = self.cache.bnds[
            self.nodes[-1].name
        ]
        if priority_mode == BranchPriorMode.MIN_LOWER_BOUND:
            return cal_priority_by_min_lower_bound(output_bound, self.verbose)
        if priority_mode == BranchPriorMode.MAX_LOWER_BOUND:
            return cal_priority_by_max_lower_bound(output_bound, self.verbose)
        if priority_mode == BranchPriorMode.MIN_DISTANCE_ZERO:
            return cal_priority_by_min_distance_zero(output_bound, self.verbose)
        if priority_mode == BranchPriorMode.MAX_DISTANCE_ZERO:
            return cal_priority_by_max_distance_zero(output_bound, self.verbose)

        raise ValueError(f"Unsupported priority mode {priority_mode}.")

    def _create_children_branches(
        self,
        priorities: list[float],
        split_input_dicts: list[SplitInputDict],
    ) -> list[Branch]:
        branches = []
        for i, (priority, split_input_dict) in enumerate(
            zip(priorities, split_input_dicts)
        ):
            new_cache = BaBPropCache(
                dtype=self.cache.dtype,
                device=self.cache.device,
                random_seed=self.cache.random_seed,
            )
            new_cache.bnds = {k: v for k, v in self.cache.bnds.items()}
            new_cache.output_final_symbnds = self.cache.output_final_symbnds

            branch = SplitInputBranch(
                self.name + str(i),
                priority,
                deepcopy(self.or_groups),
                self.nodes,
                new_cache,
                split_input_dict,
                verbose=self.verbose,
            )
            branches.append(branch)

        return branches

    def create_actual_input_bounds(
        self, original_input_bound: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        l, u = original_input_bound[0].flatten(), original_input_bound[1].flatten()
        for idx, (lb, ub) in self.cache.bnds.items():
            l[idx], u[idx] = lb, ub

        return l.clone().detach(), u.clone().detach()

    def split(
        self,
        split_mode: SplitMode = SplitMode.BISECTION,
        priority_mode: BranchPriorMode = BranchPriorMode.MAX_LOWER_BOUND,
        model: nn.Module = None,
        num_samples_for_grad_estimation: int = 1000,
    ) -> list[Branch]:

        status = analyze_status(self.each_prop_status)
        assert status != VerifStatus.UNSAFE
        if status == VerifStatus.SAFE:
            return []

        split_input_dicts = self._build_children_split_inputs(
            split_mode, model, num_samples_for_grad_estimation
        )
        priorities = self._cal_children_priorities(priority_mode)

        return self._create_children_branches(priorities, split_input_dicts)

    @property
    def split_inputs(self) -> SplitInputDict:
        return self._split_inputs
