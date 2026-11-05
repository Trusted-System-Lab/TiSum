__all__ = ["BatchedInputSplitPropModel"]

import concurrent.futures
import time
from typing import Any

import torch
from torch import Tensor

from src import BranchPriorMode
from src.args import *
from src.bab import SplitInputBranch, BranchQueue
from src.cache import BaBPropCache
from src.enums import SplitMode, PropStatus, VerifStatus
from src.model.advattack import AdvAttackModel
from src.model.inputsplitprop import (
    InputSplitPropModel,
    create_input_constrs_by_split_inputs,
)
from src.model.linprog import LinProgModel
from src.model.utils import analyze_status, analyze_each_prop_status
from src.nodes import BasicLinPropNode
from src.substorch import RelaxMode


def bound_by_linprog_model(
    self,
    model_factory: "ModelFactory",
    shared_kwargs: dict[str, Any],
    branch: SplitInputBranch,
    input_constrs: Tensor,
    output_constrs: list[Tensor],
) -> tuple[VerifStatus, Tensor | None]:
    output_bound = branch.cache.bnds[self.nodes[-1].name]

    depth = len(branch.name)
    threshold2 = torch.max(torch.clamp_max(output_bound[0], 0)).abs().item()
    threshold3 = torch.min(torch.clamp_min(output_bound[1], 0)).abs().item()
    threshold = min(threshold2, threshold3)

    is_close = threshold < self.arguments.bab_thr_to_try_linprog
    is_shallow = depth < self.arguments.bab_use_linprog_min_depth
    is_deep = depth >= self.arguments.bab_use_linprog_max_depth
    use_linprog = is_close

    is_close = threshold < self.arguments.bab_thr_to_try_mip
    is_shallow = depth < self.arguments.bab_use_mip_min_depth
    is_deep = depth >= self.arguments.bab_use_mip_max_depth
    use_mip = self.arguments.bab_use_linprog_with_mip
    use_mip = (is_close and not use_linprog) or is_deep

    if not use_linprog and not use_mip:
        return VerifStatus.UNKNOWN, None

    linprog_args = LinProgArgs(
        **shared_kwargs,
        use_mip=self.arguments.bab_use_linprog_with_mip,
        lp_timelimit=self.arguments.bab_use_linprog_timelimit,
        lp_max_num_mip_relu=self.arguments.bab_use_linprog_max_num_mip_relu,
        lp_min_range_mip_relu=self.arguments.bab_lp_min_range_mip_relu,
    )
    linprog_model: LinProgModel = model_factory.build(linprog_args)
    linprog_model.set_adv_attack_model(self._advattack_model)
    linprog_model.verbose = False
    linprog_model.cache.inherit(branch.cache)

    linprog_model.arguments.set_use_mip(use_mip)
    linprog_model.run(input_constrs, output_constrs)
    status = analyze_status(linprog_model.each_prop_status)
    adv_example = linprog_model.adv_example

    if status == VerifStatus.UNSAFE and adv_example is None and not use_mip:
        use_mip = True
        linprog_model.arguments.set_use_mip(use_mip)
        linprog_model.run(input_constrs, output_constrs)
        status = analyze_status(linprog_model.each_prop_status)
        adv_example = linprog_model.adv_example

    branch.each_prop_status = linprog_model.each_prop_status

    return status, adv_example


class BatchedInputSplitPropModel(InputSplitPropModel):
    _branch_queue: BranchQueue | None
    _models_to_run: list | None
    _advattack_model: AdvAttackModel | None
    _linprog_model: LinProgModel | None
    _start_time: float | None
    _prop_status_stats: list[dict[PropStatus, int]] | None
    _can_skip_branches_with_big_ranges: bool = True
    _global_lower_bound: Tensor | None = None
    _global_upper_bound: Tensor | None = None

    nodes: list[BasicLinPropNode]
    arguments: InputSplitPropArgs
    or_groups: list[list[int]] | None
    each_prop_status: list[list[PropStatus]] | None
    cache: BaBPropCache

    def _pre_split_branches(self, input_constrs: Tensor, max_num: int):
        if self.branch_queue.current_size >= max_num:
            return
        if self.branch_queue.empty():
            return

        input_shape = self.nodes[0].output_sizes[0]
        split_mode = SplitMode.BISECTION
        priority_mode = BranchPriorMode.MIN_LOWER_BOUND
        while self.branch_queue.current_size < max_num:
            branches = self.branch_queue.get_next_batch()
            for branch in branches:
                child_branches = branch.split(split_mode, priority_mode)
                for child_branch in child_branches:
                    child_branch: SplitInputBranch
                    input_bounds = create_input_constrs_by_split_inputs(
                        input_constrs, child_branch.split_inputs
                    )
                    child_branch.cache.bnds[self.nodes[0].name] = (
                        input_bounds[:, 0].reshape(input_shape),
                        input_bounds[:, 1].reshape(input_shape),
                    )
                self.branch_queue.put(child_branches)

                if self.branch_queue.current_size >= max_num:
                    break

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

    def _init_models_to_run(self, model_factory, shared_kwargs: dict[str, Any]):
        def check_max_size_in_model():
            max_size = 0
            for node in self.nodes:
                for output_size in node.output_sizes:
                    max_size = max(max_size, sum(output_size))
            return max_size

        args_list = []
        if self.arguments.bab_use_quadprop:
            max_size = check_max_size_in_model()
            if self.verbose:
                print("[DEBUG]: max_size =", max_size)
            if max_size > self.arguments.bab_max_size_for_quadprop:
                if self.verbose:
                    print(
                        f"Skip quadprop model because max size {max_size} > "
                        f"{self.arguments.bab_max_size_for_quadprop}."
                    )
            else:
                if self.verbose:
                    print("Initializing batched quadprop model...")
                args_list.append(
                    BatchedQuadPropArgs(
                        **shared_kwargs,
                        max_batched_size=self.arguments.bab_max_paral_branches,
                        act_relax_mode=RelaxMode.SINGLE_QUAD,
                    )
                )
        if self.arguments.bab_use_linprop:
            if self.verbose:
                print("Initializing batched linprop model...")
            args_list.append(
                BatchedLinPropArgs(
                    **shared_kwargs,
                    max_batched_size=self.arguments.bab_max_paral_branches,
                )
            )
        if self.arguments.bab_use_adaprop:
            if self.verbose:
                print("Initializing batched adaprop model.")
            args_list.append(
                BatchedAdaPropArgs(
                    **shared_kwargs,
                    max_batched_size=self.arguments.bab_max_paral_branches,
                    ada_only_last_layer=self.arguments.bab_ada_only_last_layer,
                )
            )

        if not args_list:
            raise ValueError(
                "BatchedInputSplitBaBPropModel requires at least one child model."
            )

        self._models_to_run = [model_factory.build(arg) for arg in args_list]

    def _reset_models_cache(self, branches: list[SplitInputBranch]):
        if self.verbose:
            print("Resetting child models cache...")

        batched_bounds = {}
        for node in self.nodes:
            batched_lower_bound, batched_upper_bound = [], []
            for branch in branches:
                branch_bounds = branch.cache.bnds[node.name]
                batched_lower_bound.append(branch_bounds[0])
                batched_upper_bound.append(branch_bounds[1])
            batched_lower_bound = torch.stack(batched_lower_bound, dim=0)
            batched_upper_bound = torch.stack(batched_upper_bound, dim=0)
            batched_bounds[node.name] = (batched_lower_bound, batched_upper_bound)

        def reset_model_cache(m):
            m.cache.clear()
            m.cache.bnds = batched_bounds
            if hasattr(m.cache, "output_final_symbnds"):
                m.cache.output_final_symbnds = None

        for i, model in enumerate(self._models_to_run):
            if not hasattr(model, "cache"):
                continue
            reset_model_cache(model)

        if self._linprog_model is not None:
            reset_model_cache(self._linprog_model)

    def _prepare_branch(
        self,
        branches: list[SplitInputBranch],
        input_constrs: Tensor,
        output_constrs: list[Tensor],
    ) -> tuple[list[Tensor], list[Tensor]]:
        if self.verbose:
            print("Preparing input and output constrs for branches...")
        batched_input_constrs_list: list[Tensor] = [
            create_input_constrs_by_split_inputs(input_constrs, branch.split_inputs)
            for branch in branches
        ]



        for branch in branches:
            if self.arguments.bab_split_mode == SplitMode.GRAD:
                branch.set_output_constrs(output_constrs)

        return batched_input_constrs_list, output_constrs

    def _bound_by_batched_models_to_run(
        self,
        branches: list[SplitInputBranch],
        batched_branch_input_constrs: Tensor,
        output_constrs: list[Tensor],
    ) -> tuple[list[VerifStatus], list[Tensor | None]]:
        time_start = time.perf_counter()
        if self.verbose:
            print(f"Bounding by {len(self._models_to_run)} child models...")

        batched_statuses: list[VerifStatus] = []
        batched_adv_examples: list[Tensor | None] = []
        models = self._models_to_run
        model = None

        batch_size = len(branches)
        if self.verbose:
            print(f"Set batched model batch size to {batch_size}.")
        for model in models:
            model.set_batch_size(batch_size)

        from src.verifier import batched_run_models_with_same_bounds

        batched_run_models_with_same_bounds(
            self._models_to_run,
            input_constrs=batched_branch_input_constrs,
            output_constrs=output_constrs,
        )

        batched_each_prop_status = model.each_prop_status
        n_groups = len(self.or_groups)
        start = 0
        end = start + n_groups
        for i, branch in enumerate(branches):
            branch.each_prop_status = batched_each_prop_status[start:end]
            status = analyze_status(branch.each_prop_status)
            batched_statuses.append(status)
            batched_adv_examples.append(None)
            start = end
            end = start + n_groups

        batched_bounds = model.cache.bnds
        batched_output_final_symbnds = model.cache.output_final_symbnds
        for i, branch in enumerate(branches):
            for node in self.nodes:
                branch_lower_bound = batched_bounds[node.name][0][i]
                branch_upper_bound = batched_bounds[node.name][1][i]
                branch.cache.bnds[node.name] = (branch_lower_bound, branch_upper_bound)

            branch.cache.output_final_symbnds = tuple(
                batched_output_final_symbnds[j][i] for j in range(4)
            )

            if self.verbose:
                from src.utils import str_list

                output_bounds = branch.cache.bnds[self.nodes[-1].name]
                if self.verbose and len(output_bounds[0]) < 20:
                    print(
                        f"{branch}, "
                        f"Each status: {branch.each_prop_status}, "
                        f"Adv example: False, "
                        f"Lower bound: {str_list(output_bounds[0].tolist())}, "
                        f"Upper bound: {str_list(output_bounds[1].tolist())}"
                    )
        time_end = time.perf_counter()
        if self.verbose:
            num_safe = sum(1 for s in batched_statuses if s == VerifStatus.SAFE)
            num_unsafe = sum(1 for s in batched_statuses if s == VerifStatus.UNSAFE)
            print(
                f"Bounded {len(branches)} branches by batched models in "
                f"{time_end - time_start:.4f}s\n"
                f"Results: SAFE/UNSAFE/TOTAL = {num_safe}/{num_unsafe}/{len(branches)}"
            )
        return batched_statuses, batched_adv_examples

    def _bound_by_child_multithread_linprog_model(
        self,
        branches: list[SplitInputBranch],
        batched_input_constrs: Tensor,
        output_constrs: list[Tensor],
        batched_statuses: list[VerifStatus],
        batched_adv_examples: list[Tensor | None],
    ) -> tuple[list[VerifStatus], list[Tensor | None]]:
        unverified_indices = [
            i
            for i, status in enumerate(batched_statuses)
            if status == VerifStatus.UNKNOWN
        ]
        if not unverified_indices:
            return batched_statuses, batched_adv_examples

        branches = [branches[i] for i in unverified_indices]
        batched_input_constrs = batched_input_constrs[unverified_indices]
        batched_input_constrs = [
            batched_input_constrs[i] for i in range(len(unverified_indices))
        ]

        args = self.arguments
        shared_kwargs = {
            "use_vnnlib": args.use_vnnlib,
            "onnx_path": args.onnx_path,
            "onnx_model": args.onnx_model,
            "slimonnx_path": args.slimonnx_path,
            "torchonnx_path": args.torchonnx_path,
            "torchonnx_model": args.torchonnx_model,
            "torchonnx_class_name": args.torchonnx_class_name,
            "torchonnx_param_path": args.torchonnx_param_path,
            "store_all_bnds": True,
            "store_output_final_symbnds": True,
            "random_seed": args.random_seed,
            "dtype": args.dtype,
            "device": args.device,
        }

        from src.factory import ModelFactory

        model_factory = ModelFactory()
        max_workers = self.arguments.bab_max_paral_branches
        time_start = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(
                    bound_by_linprog_model,
                    self,
                    model_factory,
                    shared_kwargs,
                    branch,
                    input_constrs,
                    output_constrs,
                ): i
                for i, (branch, input_constrs) in enumerate(
                    zip(branches, batched_input_constrs)
                )
            }
            for future in concurrent.futures.as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    status, adv_example = future.result()
                except Exception as e:
                    raise e
                    print(f"LinProg model generated an exception: {e}")
                    status, adv_example = VerifStatus.UNKNOWN, None
                batched_statuses[unverified_indices[i]] = status
                batched_adv_examples[unverified_indices[i]] = adv_example

        time_end = time.perf_counter()
        if self.verbose:
            num_safe = sum(1 for s in batched_statuses if s == VerifStatus.SAFE)
            num_unsafe = sum(1 for s in batched_statuses if s == VerifStatus.UNSAFE)
            print(
                f"Bounded {len(branches)} branches by linprog models in "
                f"{time_end - time_start:.4f}s\n"
                f"Results: SAFE/UNSAFE/TOTAL = "
                f"{num_safe}/{num_unsafe}/{len(batched_statuses)}"
            )

        return batched_statuses, batched_adv_examples

    def bound(
        self,
        branches: list[SplitInputBranch],
        input_constrs: Tensor,
        output_constrs: list[Tensor],
    ):
        UNKNOWN, SAFE, UNSAFE = (
            VerifStatus.UNKNOWN,
            VerifStatus.SAFE,
            VerifStatus.UNSAFE,
        )
        status = UNKNOWN
        n_branches = len(branches)

        if self.verbose:
            print(f"Bounding {n_branches} branches...")
        for branch in branches:
            self._add_child_node_to_branching_tree(branch)

        batched_input_constrs_list, output_constrs = self._prepare_branch(
            branches, input_constrs, output_constrs
        )
        batched_input_constrs = torch.stack(batched_input_constrs_list, dim=0)

        self._reset_models_cache(branches)
        if self.verbose:
            print("Start Batched BaB bounding by child models.")
        batched_statuses, batched_adv_examples = self._bound_by_batched_models_to_run(
            branches, batched_input_constrs, output_constrs
        )

        if self.arguments.bab_use_linprog_when_small_range:
            batched_statuses, batched_adv_examples = (
                self._bound_by_child_multithread_linprog_model(
                    branches,
                    batched_input_constrs,
                    output_constrs,
                    batched_statuses,
                    batched_adv_examples,
                )
            )

        time_start = time.perf_counter()
        n_attacks = 0
        branch = None
        for i, (branch, status, adv_example) in enumerate(
            zip(branches, batched_statuses, batched_adv_examples)
        ):
            branch_input_constrs = batched_input_constrs_list[i]

            output_bound = branch.cache.bnds[self.nodes[-1].name]
            threshold = torch.min(torch.clamp(output_bound[1], min=0.0)).abs().item()
            is_small = threshold < self.arguments.bab_thr_to_try_adv_attack
            if (status == UNKNOWN and is_small) or (
                status == UNSAFE and adv_example is None
            ):
                if self.verbose:
                    print(f"Running post adv attack...")
                status, adv_example = self._post_adv_attack(
                    branch, branch_input_constrs, output_constrs
                )
                n_attacks += 1

            self._update_parent_node_in_branching_tree(branch)
            n_branches -= 1

            batched_statuses[i] = status
            batched_adv_examples[i] = adv_example
            if status == UNSAFE and adv_example is not None:
                if self.verbose:
                    print(f"Find adv example(s) for {branch}.")
                self._adv_example = adv_example
                break

            if time.perf_counter() - self._time_start > self.arguments.bab_timeout:
                if self.verbose:
                    print(
                        f"BaB bounding timed out, stop BaB bounding with remaining "
                        f"{n_branches} branches."
                    )
                break

        time_end = time.perf_counter()
        if self.verbose:
            num_safe = sum(1 for s in batched_statuses if s == VerifStatus.SAFE)
            num_unsafe = sum(1 for s in batched_statuses if s == VerifStatus.UNSAFE)
            print(
                f"Try to attack {n_attacks} branches by post adv attack models in "
                f"{time_end - time_start:.4f}s\n"
                f"Results: SAFE/UNSAFE/TOTAL = "
                f"{num_safe}/{num_unsafe}/{len(batched_statuses)}"
            )

        self._cal_global_bounds()

        if status == UNSAFE:
            self._each_prop_status = branch.each_prop_status
            return

        global_bounds: tuple[Tensor, Tensor] = (
            self._global_lower_bound,
            self._global_upper_bound,
        )

        self.each_prop_status = analyze_each_prop_status(global_bounds, self.or_groups)

    def run(self, input_constrs: Tensor, output_constrs: list[Tensor]):
        self._time_start = time.perf_counter()

        self._check_cache()
        self.set_input_constrs(input_constrs)
        self.set_output_constrs(output_constrs)
        self._init_child_models()
        self._init_branch_queue(input_constrs, output_constrs)

        self._pre_split_branches(input_constrs, self.arguments.bab_max_paral_branches)

        dtype, device = self.cache.dtype, self.cache.device
        input_constrs = input_constrs.to(dtype=dtype, device=device)
        output_constrs = [c.to(dtype=dtype, device=device) for c in output_constrs]

        branch_counter = [self.branch_queue.current_size]

        while True:

            self._print_current_stats(branch_counter[0])

            branches: list[SplitInputBranch] = (
                self.branch_queue.get_next_batch()
            )
            self.bound(branches, input_constrs, output_constrs)
            self._record_prop_status_stats(branches)
            if self._if_early_stop(branches):
                break

            self.branch(branches, output_constrs, branch_counter)
            if self._if_early_stop(branches) or self._if_safe():
                break

        self._print_final_stats(branch_counter[0])
        self._check_prop_status_stats()
