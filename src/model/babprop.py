__all__ = ["BaBPropModel"]

import time
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy

import torch
from torch import Tensor

from src.args import *
from src.bab import (
    Branch,
    BranchQueue,
    SplitInputBranch,
    SplitInputDict,
)
from src.cache import BaBPropCache
from src.enums import PropStatus, VerifStatus
from src.model.advattack import AdvAttackModel
from src.model.adaprop import AdaPropModel
from src.model.linprog import LinProgModel
from src.nodes import (
    BasicLinPropNode,
    LinearLinPropNode,
    NonlinearLinPropNode,
    MaxPool2dNode,
)


class BaBPropModel(AdaPropModel, ABC):
    """This is a father class for BaB-based propagation models."""

    _branch_queue: BranchQueue | None
    _models_to_run: list | None
    _advattack_model: AdvAttackModel | None
    _linprog_model: LinProgModel | None
    _start_time: float | None
    _prop_status_stats: list[dict[PropStatus, int]] | None
    _global_lower_bound: Tensor | None = None
    _global_upper_bound: Tensor | None = None

    nodes: list[BasicLinPropNode]
    arguments: BaBPropArgs
    or_groups: list[list[int]] | None
    each_prop_status: list[list[PropStatus]] | None
    cache: BaBPropCache

    def __init__(self, nodes, arguments: BaBPropArgs, verbose=False):
        super().__init__(nodes, arguments, verbose)
        self._branch_queue = None
        self._models_to_run = None
        self._advattack_model = None
        self._linprog_model = None

        self._live_branches_counter = None
        self._front_branches = None
        self._global_status = VerifStatus.UNKNOWN
        self._global_lower_bound = None
        self._global_upper_bound = None

        self._prop_status_stats = None

        self._time_start = None

    def _check_cache(self):
        if len(self.cache.symbnds) != 0:
            raise RuntimeError(
                f"There should be no cached symbolic bounds but "
                f"{len(self.cache.symbnds)}."
            )

        for node in self.nodes[:-1]:
            if isinstance(node, LinearLinPropNode):
                if node.name not in self.cache.bnds:
                    raise RuntimeError(
                        f"Linear node {node.name} has no bounds in the cache. "
                        f"Please call a normal bound propagation first."
                    )

        if isinstance(self.arguments, InputSplitPropArgs):
            return

        for node in self.nodes:
            if not isinstance(node, NonlinearLinPropNode):
                continue

            if node.name not in self.cache.rlxs:
                raise RuntimeError(
                    f"BaB propagation needs the relaxation of {node.name}. "
                    f"Please call a normal bound propagation first."
                )
            rlx = self.cache.rlxs[node.name]
            (
                lower_rlx_coeffs,
                lower_rlx_const,
                upper_rlx_coeffs,
                upper_rlx_const,
            ) = rlx
            if not isinstance(node, MaxPool2dNode) and (
                lower_rlx_coeffs.dim() in {1, 2}
                or lower_rlx_const.dim() != 1
                or upper_rlx_coeffs.dim() <= 2
                or upper_rlx_const.dim() in {1, 2}
            ):
                raise RuntimeError(
                    f"Relaxation of {node.name} has invalid shape: "
                    f"{lower_rlx_coeffs.shape}, {lower_rlx_const.shape}, "
                    f"{upper_rlx_coeffs.shape}, {upper_rlx_const.shape}. "
                    f"Currently, only support the 1d/2d coeffs and 1d const for "
                    f"NonlinearNode excluding MaxPool. "
                    f"This means the relaxation can only be single-neuron linear "
                    f"relaxation. "
                )
            elif isinstance(node, MaxPool2dNode) and (
                lower_rlx_coeffs.dim() in {2, 3}
                or lower_rlx_const.dim() != 1
                or upper_rlx_coeffs.dim() in {2, 3}
                or upper_rlx_const.dim() != 1
            ):
                raise RuntimeError(
                    f"Relaxation of {node.name} has invalid shape: "
                    f"{lower_rlx_coeffs.shape}, {lower_rlx_const.shape}, "
                    f"{upper_rlx_coeffs.shape}, {upper_rlx_const.shape}. "
                    f"Currently, only support the 2d/3d for coeffs and 1d for "
                    f"const for MaxPool2d node. "
                    f"This means the relaxation can only be single-neuron linear "
                    f"relaxation."
                )

            if any(isinstance(node, NonlinearLinPropNode) for node in node.next_nodes):
                if node.name not in self.cache.bnds:
                    raise RuntimeError(
                        f"Split neuron mode needs the bnd of {node.name}. "
                        f"Please call a normal bound propagation first and "
                        f"set argument store_all_bnds=True of BasicArgs."
                    )

    def set_output_constrs(self, output_constrs: list[Tensor]):
        super().set_output_constrs(output_constrs)

        self._global_status = VerifStatus.UNKNOWN
        n = sum(g.size(0) for g in output_constrs)
        self._global_lower_bound = torch.full(
            (n,), torch.inf, dtype=self.cache.dtype, device=self.cache.device
        )
        self._global_upper_bound = torch.full(
            (n,), -torch.inf, dtype=self.cache.dtype, device=self.cache.device
        )

        template_dict = {status: 0 for status in PropStatus}
        self._prop_status_stats = [
            deepcopy(template_dict) for g in output_constrs for _ in range(g.size(0))
        ]

    def _cal_global_bounds(self):
        global_lb = torch.full_like(self._global_lower_bound, torch.inf)
        global_ub = torch.full_like(self._global_upper_bound, -torch.inf)
        for branch in self._front_branches.values():
            branch_bound = branch.cache.bnds[self.nodes[-1].name]
            branch_lb = branch_bound[0]
            branch_ub = branch_bound[1]
            global_lb = torch.minimum(global_lb, branch_lb)
            global_ub = torch.maximum(global_ub, branch_ub)

        if self.verbose:
            if len(global_lb) < 20:
                print(
                    f"Global bounds ({len(self._front_branches)} branches): "
                    f"Lower bound: {global_lb.tolist()}, "
                    f"Upper bound: {global_ub.tolist()}"
                )

        self._global_lower_bound = global_lb
        self._global_upper_bound = global_ub

        output_node: OutputNode = self.nodes[-1]
        self.cache.bnds[output_node.name] = (global_lb, global_ub)

    @abstractmethod
    def _init_child_models(self):
        pass

    def _build_branch_cache(self):
        branch_cache = BaBPropCache(
            dtype=self.cache.dtype,
            device=self.cache.device,
            random_seed=self.cache.random_seed,
        )
        branch_cache.bnds = {
            k: (v[0].clone().detach(), v[1].clone().detach())
            for k, v in self.cache.bnds.items()
        }
        if self.cache.output_final_symbnds is not None:
            branch_cache.output_final_symbnds = (
                self.cache.output_final_symbnds[0].clone().detach(),
                self.cache.output_final_symbnds[1].clone().detach(),
            )

        return branch_cache

    def _build_split_dict(self):
        return SplitInputDict()

    def _build_branch(self):
        return SplitInputBranch

    def _build_root_branch(self) -> Branch:
        if self.verbose:
            print("Building root branch...")
        branch_cache = self._build_branch_cache()
        split_dict = self._build_split_dict()
        _Branch = self._build_branch()
        return _Branch(
            "0",
            0,
            deepcopy(self.or_groups),
            self.nodes,
            branch_cache,
            split_dict,
            verbose=self.verbose,
        )

    def _init_branch_queue(self, input_constrs: Tensor, output_constrs: list[Tensor]):
        if self.verbose:
            print("Initializing branch queue....")

        self._branch_queue = BranchQueue(
            self.arguments.bab_max_cached_branches,
            self.arguments.bab_max_paral_branches,
        )
        root_branch = self._build_root_branch()
        self.branch_queue.put(root_branch)
        self._front_branches = {}
        self._live_branches_counter = {}

    @abstractmethod
    def bound(
        self,
        branches: list[Branch],
        input_constrs: Tensor,
        output_constrs: list[Tensor],
    ):
        pass

    @abstractmethod
    def branch(
        self,
        branches: list[Branch],
        output_constrs: list[Tensor],
        branches_counter: list[int],
    ):
        pass

    def _record_prop_status_stats(self, branches: list[Branch]):
        for branch in branches:
            i = 0
            for status_group in branch.each_prop_status:
                for status in status_group:
                    self._prop_status_stats[i][status] += 1
                    i += 1

    def _if_early_stop(self, branches: list[Branch]) -> bool:
        if self._adv_example is not None:
            self._global_status = VerifStatus.UNSAFE
            if self.verbose:
                print("Found an unsatisfiable branch, stop BaB search.")
            return True

        if time.perf_counter() - self._time_start > self.arguments.bab_timeout:
            self._global_status = VerifStatus.TIMEOUT
            if self.verbose:
                print("BaB search timed out, stop BaB search.")
            return True

        if 0 < self.arguments.bab_max_cached_branches <= self.branch_queue.current_size:
            self._global_status = VerifStatus.MEMOUT
            if self.verbose:
                print(
                    f"Branch number exceeded {self.arguments.bab_max_cached_branches}, "
                    f"stop BaB search."
                )
            return True

        if len(branches[-1].name) > self.arguments.bab_max_depth:
            self._global_status = VerifStatus.MEMOUT
            if self.verbose:
                print(
                    f"Branch {branches[0].name} is too deep, "
                    f"skip and continue with the next batch."
                )
            return True

        return False

    def _if_safe(self) -> bool:
        if self.branch_queue.current_size == 0:
            for prop_group in self.each_prop_status:
                for k in range(len(prop_group)):
                    prop_group[k] = PropStatus.SAFE
            self._global_status = VerifStatus.SAFE

            if self.verbose:
                print("No more branches to process, stop BaB search.")
            return True
        return False

    def _print_current_stats(self, total_n_branches: int):
        if self.verbose:
            print(
                f"\nCurrent/Total branches: "
                f"{self.branch_queue.current_size}/{total_n_branches}"
            )

    def _print_final_stats(self, total_n_branches: int):
        if self.verbose:
            print(
                f"Total {total_n_branches} branches processed, "
                f"final branch queue size: {self.branch_queue.current_size}"
            )
            print("Property status statistics:")
            for i, status in enumerate(self._prop_status_stats):
                print(f"Property {i}:")
                for k, v in status.items():
                    print(f"\t{str(k):<8}: {v:>3}")

    def _check_prop_status_stats(self):
        has_unsat = False
        all_has_sat = [False] * len(self._prop_status_stats)
        for i, status in enumerate(self._prop_status_stats):
            if status[PropStatus.UNSAFE] > 0:
                has_unsat = True
            if status[PropStatus.SAFE] > 0:
                all_has_sat[i] = True
        if not has_unsat:
            if not all(all_has_sat):
                warnings.warn(
                    "No unsatisfiable branch found, but some output "
                    "constraints are not satisfied. This may indicate that the model "
                    "is not correctly set up or the arguments are not properly "
                    "configured (like optimization hyperparameters). We will treat the "
                    "final result as UNKNOWN.",
                    UserWarning,
                )
                self._global_status = VerifStatus.UNKNOWN

    def _add_child_node_to_branching_tree(self, branch: Branch):
        self._front_branches[branch.name] = branch
        self._live_branches_counter[branch.name] = 0

    def _update_parent_node_in_branching_tree(self, branch: Branch):
        parent_branch_name = branch.name[:-1]
        if len(parent_branch_name) != 0:
            if parent_branch_name not in self._live_branches_counter:
                return
            self._live_branches_counter[parent_branch_name] += 1
            if self._live_branches_counter[parent_branch_name] == 2:
                del self._front_branches[parent_branch_name]

    def run(self, input_constrs: Tensor, output_constrs: list[Tensor]):
        self._time_start = time.perf_counter()

        self._check_cache()
        self.set_input_constrs(input_constrs)
        self.set_output_constrs(output_constrs)
        self._init_child_models()
        self._init_branch_queue(input_constrs, output_constrs)

        dtype, device = self.cache.dtype, self.cache.device
        input_constrs = input_constrs.to(dtype=dtype, device=device)
        output_constrs = [c.to(dtype=dtype, device=device) for c in output_constrs]

        branch_counter = [self.branch_queue.current_size]

        while True:

            self._print_current_stats(branch_counter[0])

            branches = self.branch_queue.get_next_batch()
            self.bound(branches, input_constrs, output_constrs)
            self._record_prop_status_stats(branches)
            if self._if_early_stop(branches):
                break

            self.branch(branches, output_constrs, branch_counter)
            if self._if_early_stop(branches) or self._if_safe():
                break

        self._print_final_stats(branch_counter[0])
        self._check_prop_status_stats()

    @property
    def branch_queue(self) -> BranchQueue:
        return self._branch_queue
