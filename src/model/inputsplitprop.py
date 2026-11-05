__all__ = ["InputSplitPropModel", "create_input_constrs_by_split_inputs"]

import time
from typing import Any

import torch
from torch import Tensor

from src import str_list
from src.args import *
from src.bab import BranchQueue, SplitInputDict, SplitInputBranch
from src.cache import BaBPropCache
from src.enums import PropStatus, VerifStatus, SplitMode
from src.model.advattack import AdvAttackModel
from src.model.babprop import BaBPropModel
from src.model.linprog import LinProgModel
from src.model.utils import analyze_each_prop_status, analyze_status
from src.nodes import BasicLinPropNode
from src.substorch import RelaxMode, cal_scalar_bound


def create_input_constrs_by_split_inputs(
    input_constrs: Tensor, split_inputs: SplitInputDict | None
) -> Tensor:
    assert input_constrs.dim() == 2
    assert input_constrs.size(1) == 2

    if not split_inputs:
        return input_constrs

    input_constrs = input_constrs.clone()
    for idx, (l, u) in split_inputs.items():
        input_constrs[idx, 0], input_constrs[idx, 1] = l, u

    return input_constrs.detach()


class InputSplitPropModel(BaBPropModel):
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

    def _init_advattack_model(self, model_factory, shared_kwargs: dict[str, Any]):
        self._advattack_model = None
        if self.arguments.bab_use_adv_attack_before_prop:
            if self.verbose:
                print("Initializing advattack model...")
            adv_attack_args = AdvAttackArgs(
                **shared_kwargs,
                adv_lr=self.arguments.bab_adv_lr,
                adv_min_loss_change=self.arguments.bab_adv_min_loss_change,
                adv_restarts=self.arguments.bab_adv_restarts,
                adv_num_points=self.arguments.bab_adv_num_points,
                adv_max_iters=self.arguments.bab_adv_max_iters,
            )
            self._advattack_model = model_factory.build(adv_attack_args)

    def _init_linprog_model(self, model_factory, shared_kwargs: dict[str, Any]):
        self._linprog_model = None
        if self.arguments.bab_use_linprog_when_small_range:
            if self.verbose:
                print("Initializing linprog model...")
            linprog_args = LinProgArgs(
                **shared_kwargs,
                use_mip=self.arguments.bab_use_linprog_with_mip,
                lp_timelimit=self.arguments.bab_use_linprog_timelimit,
                lp_max_num_mip_relu=self.arguments.bab_use_linprog_max_num_mip_relu,
                lp_min_range_mip_relu=self.arguments.bab_lp_min_range_mip_relu,
            )
            self._linprog_model = model_factory.build(linprog_args)
            self._linprog_model.set_adv_attack_model(self._advattack_model)

    def _init_models_to_run(self, model_factory, shared_kwargs: dict[str, Any]):
        args_list = []
        if self.arguments.bab_use_quadprop:
            if self.verbose:
                print("Initializing quadprop model...")
            args_list.append(
                QuadPropArgs(
                    **shared_kwargs,
                    act_relax_mode=RelaxMode.SINGLE_QUAD,
                )
            )
        if self.arguments.bab_use_linprop:
            if self.verbose:
                print("Initializing linprop model...")
            args_list.append(
                LinPropArgs(
                    **shared_kwargs,
                )
            )
        if self.arguments.bab_use_adaprop:
            if self.verbose:
                print("Initializing adaprop model.")
            args_list.append(
                AdaPropArgs(
                    **shared_kwargs,
                    ada_only_last_layer=self.arguments.bab_ada_only_last_layer,
                )
            )

        if not args_list:
            raise ValueError(
                "InputSplitBaBPropModel requires at least one child model."
            )

        self._models_to_run = [model_factory.build(arg) for arg in args_list]

    def _init_child_models(self):
        assert not self.arguments.use_dataset
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
        model_factory.onnx_model = args.onnx_model
        model_factory.torchonnx_model = args.torchonnx_model

        self._init_advattack_model(model_factory, shared_kwargs)
        self._init_linprog_model(model_factory, shared_kwargs)
        self._init_models_to_run(model_factory, shared_kwargs)

    def _reset_models_cache(self, branch: SplitInputBranch):
        if self.verbose:
            print("Resetting child models' cache...")

        def reset_model_cache(m):
            m.cache.clear()
            m.cache.inherit(branch.cache)

        for i, model in enumerate(self._models_to_run):
            if not hasattr(model, "cache"):
                continue
            reset_model_cache(model)

        if self._linprog_model is not None:
            reset_model_cache(self._linprog_model)

        for m in self._models_to_run + [self._linprog_model]:
            if hasattr(m, "cache"):
                assert id(m.cache.bnds) == id(branch.cache.bnds)

    def _prepare_branch(
        self,
        branch: SplitInputBranch,
        input_constrs: Tensor,
        output_constrs: list[Tensor],
    ) -> tuple[Tensor, list[Tensor]]:
        if self.verbose:
            print("Preparing input and output constrs for branches...")

        branch_input_constrs = create_input_constrs_by_split_inputs(
            input_constrs, branch.split_inputs
        )

        if self.arguments.bab_split_mode == SplitMode.GRAD:
            branch.set_output_constrs(output_constrs)

        return branch_input_constrs, output_constrs

    def _bound_by_output_symbnds(
        self, branch: SplitInputBranch, input_constrs: Tensor
    ) -> tuple[VerifStatus, None, bool]:
        if self.verbose:
            print(f"Bounding by final symbolic bounds...")

        status = VerifStatus.UNKNOWN
        if (
            not self.arguments.bab_use_direct_symbols
            or branch.cache.output_final_symbnds is None
        ):
            if self.verbose:
                print(
                    f"Skip bounding by final symbolic bounds due to not set or disabled."
                )
            return status, None, False

        if (
            self._branch_queue.current_size
            > self.arguments.bab_max_cached_branches_for_only_split
        ):
            self._can_skip_branches_with_big_ranges = False
        else:
            self._can_skip_branches_with_big_ranges = True
        if self.verbose and not self._can_skip_branches_with_big_ranges:
            print(
                f"Too many ({self._branch_queue.current_size}) branches in the queue, "
                f"do not directly split {branch}."
            )
            return status, None, False

        symbnds = branch.cache.output_final_symbnds
        assert symbnds is not None
        input_shape = symbnds[0].shape[1:]
        input_lower = input_constrs[:, 0].reshape(input_shape)
        input_upper = input_constrs[:, 1].reshape(input_shape)
        input_bounds = (input_lower, input_upper)
        output_bounds = cal_scalar_bound(*symbnds, *input_bounds)
        branch.cache.bnds[self.nodes[-1].name] = output_bounds
        each_prop_status = analyze_each_prop_status(output_bounds, self.or_groups)
        branch.update_each_prop_status(each_prop_status)
        status = analyze_status(each_prop_status)

        depth = len(branch.name)
        threshold = torch.min(output_bounds[1] - output_bounds[0]).item()
        un_solved = status == VerifStatus.UNKNOWN
        is_shallow = depth <= self.arguments.bab_max_depth_for_only_split
        is_large_range = threshold >= self.arguments.bab_max_range_for_only_split
        is_small_range = threshold <= self.arguments.bab_min_range_for_only_split
        split = un_solved and ((is_shallow and is_large_range) or is_small_range)

        if self.verbose and split:
            print(f"Continue split {branch} with {status}.")
        if self.verbose and len(output_bounds[0]) < 20:
            print(
                f"Status: {status}, "
                f"Each status: {branch.each_prop_status}, "
                f"Adv example: False, "
                f"Lower bound: {str_list(output_bounds[0].tolist())}, "
                f"Upper bound: {str_list(output_bounds[1].tolist())}"
            )

        return status, None, split

    def _bound_by_models_to_run(
        self,
        branch: SplitInputBranch,
        input_constrs: Tensor,
        output_constrs: list[Tensor],
    ) -> tuple[VerifStatus, Tensor | None]:

        if self.verbose:
            print(f"Bounding by {len(self._models_to_run)} child models...")

        from src.verifier import run_models_with_same_bounds

        status, adv_example = run_models_with_same_bounds(
            self._models_to_run,
            input_constrs=input_constrs,
            output_constrs=output_constrs,
        )
        branch.update_each_prop_status(self._models_to_run[-1].each_prop_status)

        for m in self._models_to_run:
            if hasattr(m, "cache"):
                assert id(m.cache.bnds) == id(branch.cache.bnds)
                if m.cache.output_final_symbnds is not None:
                    branch.cache.output_final_symbnds = m.cache.output_final_symbnds

        if self.verbose:
            output_bounds = branch.cache.bnds[self.nodes[-1].name]
            if self.verbose and len(output_bounds[0]) < 20:
                print(
                    f"Status: {status}, "
                    f"Each status: {branch.each_prop_status}, "
                    f"Adv example: False, "
                    f"Lower bound: {str_list(output_bounds[0].tolist())}, "
                    f"Upper bound: {str_list(output_bounds[1].tolist())}"
                )

        return status, adv_example

    def _bound_by_linprog_model(
        self,
        branch: SplitInputBranch,
        input_constrs: Tensor,
        output_constrs: list[Tensor],
    ) -> tuple[VerifStatus, Tensor | None]:
        if self._linprog_model is None:
            return VerifStatus.UNKNOWN, None
        if self.verbose:
            print(f"Bounding by LP model...")

        linprog_model = self._linprog_model
        linprog_model.cache.inherit(branch.cache)
        output_bound = branch.cache.bnds[self.nodes[-1].name]

        depth = len(branch.name)
        threshold = torch.max(torch.clamp(output_bound[0], max=0.0)).abs().item()
        is_close = threshold < self.arguments.bab_thr_to_try_linprog
        is_shallow = depth < self.arguments.bab_use_linprog_min_depth
        is_deep = depth >= self.arguments.bab_use_linprog_max_depth
        use_linprog = (is_close or is_deep) and not is_shallow

        is_close = threshold < self.arguments.bab_thr_to_try_mip
        is_shallow = depth < self.arguments.bab_use_mip_min_depth
        is_deep = depth >= self.arguments.bab_use_mip_max_depth
        use_mip = self.arguments.bab_use_linprog_with_mip
        use_mip = use_mip and (is_close or is_deep) and not use_linprog

        if not use_linprog and not use_mip:
            if self.verbose:
                print(f"Skip LP due to not satisfy the conditions.")
            return VerifStatus.UNKNOWN, None

        if self.verbose:
            print(f"Running {'MIP' if use_mip else 'LP'}...")

        linprog_model.arguments.set_use_mip(use_mip)
        linprog_model.run(input_constrs, output_constrs)
        status = analyze_status(linprog_model.each_prop_status)
        adv_example = linprog_model.adv_example

        if not use_mip and status == VerifStatus.UNSAFE and adv_example is None:
            use_mip = True
            if self.verbose:
                print(f"Result is UNSAFE without adv example. Try MIP to find.")
            linprog_model.arguments.set_use_mip(use_mip)
            linprog_model.run(input_constrs, output_constrs)
            status = analyze_status(linprog_model.each_prop_status)
            adv_example = linprog_model.adv_example

        assert id(linprog_model.cache.bnds) == id(branch.cache.bnds)
        branch.update_each_prop_status(linprog_model.each_prop_status)

        if self.verbose:
            output_bounds = branch.cache.bnds[self.nodes[-1].name]
            if self.verbose and len(output_bounds[0]) < 20:
                print(
                    f"Status: {status}, "
                    f"Each status: {branch.each_prop_status}, "
                    f"Adv example: False, "
                    f"Lower bound: {str_list(output_bounds[0].tolist())}, "
                    f"Upper bound: {str_list(output_bounds[1].tolist())}"
                )

        return status, adv_example

    def _post_adv_attack(
        self,
        branch: SplitInputBranch,
        input_constrs: Tensor,
        output_constrs: list[Tensor],
    ) -> tuple[VerifStatus, Tensor | None]:
        attack_model = self._advattack_model
        attack_model.run(input_constrs, output_constrs)
        status = analyze_status(attack_model.each_prop_status)
        adv_example = attack_model.adv_example

        if status == VerifStatus.UNKNOWN:
            for i in range(len(branch.each_prop_status)):
                for j in range(len(branch.each_prop_status[i])):
                    if branch.each_prop_status[i][j] == PropStatus.UNSAFE:
                        branch.each_prop_status[i][j] = PropStatus.UNKNOWN

        return status, adv_example

    def bound(
        self,
        branches: list[SplitInputBranch],
        input_constrs: Tensor,
        output_constrs: list[Tensor],
    ):
        SAFE, UNSAFE, UNKNOWN = (
            VerifStatus.SAFE,
            VerifStatus.UNSAFE,
            VerifStatus.UNKNOWN,
        )
        status = UNKNOWN

        def update_result(
            new_result: tuple[VerifStatus, Tensor | None],
            old_result: tuple[VerifStatus, Tensor | None],
        ) -> tuple[VerifStatus, Tensor | None]:
            new_status, new_adv_example = new_result
            old_status, old_adv_example = old_result

            if new_status in {SAFE, UNSAFE}:
                assert old_status == UNKNOWN or old_status == new_status
                old_status = new_status
            if new_adv_example is not None:
                old_adv_example = new_adv_example
            return old_status, old_adv_example

        branch = None
        n_branches = len(branches)
        for branch in branches:
            if self.verbose:
                print(f"Bounding {branch} (depth={len(branch.name)})...")

            self._add_child_node_to_branching_tree(branch)

            branch_input_constrs, branch_output_constrs = self._prepare_branch(
                branch, input_constrs, output_constrs
            )

            self._reset_models_cache(branch)

            status, adv_example, continue_split = self._bound_by_output_symbnds(
                branch, branch_input_constrs
            )
            result = (status, adv_example)

            if not continue_split:
                if status == UNKNOWN:
                    new_result = self._bound_by_models_to_run(
                        branch, branch_input_constrs, branch_output_constrs
                    )
                    result = update_result(new_result, result)
                    status, adv_example = result

                if status == UNKNOWN or (status == UNSAFE and adv_example is None):
                    new_result = self._bound_by_linprog_model(
                        branch, branch_input_constrs, branch_output_constrs
                    )
                    result = update_result(new_result, result)
                    status, adv_example = result

                output_bound = branch.cache.bnds[self.nodes[-1].name]
                thr = torch.min(torch.clamp(output_bound[1], min=0.0)).abs().item()
                is_small = thr < self.arguments.bab_thr_to_try_adv_attack

                if (status == UNKNOWN and is_small) or (
                    status == UNSAFE and adv_example is None
                ):
                    if self.verbose:
                        print(f"Running post adv attack...")
                    new_result = self._post_adv_attack(
                        branch, branch_input_constrs, branch_output_constrs
                    )
                    result = update_result(new_result, result)
                    status, adv_example = result

            self._update_parent_node_in_branching_tree(branch)
            n_branches -= 1

            if status == UNSAFE and adv_example is not None:
                if self.verbose:
                    print(f"Find adv example(s).")
                self._adv_example = adv_example
                break

            if time.perf_counter() - self._time_start > self.arguments.bab_timeout:
                if self.verbose:
                    print(
                        f"BaB bounding timed out, stop with remaining "
                        f"{n_branches} branches."
                    )
                break

        self._cal_global_bounds()

        if status == UNSAFE:
            self._each_prop_status = branch.each_prop_status
            return

        global_bounds: tuple[Tensor, Tensor] = (
            self._global_lower_bound,
            self._global_upper_bound,
        )
        self.each_prop_status = analyze_each_prop_status(global_bounds, self.or_groups)

    def branch(
        self,
        branches: list[SplitInputBranch],
        output_constrs: list[Tensor],
        branches_counter: list[int],
    ):
        if self.verbose:
            print(f"Branching {len(branches)} branches.")

        for branch in branches:
            new_branches = branch.split(
                self.arguments.bab_split_mode,
                self.arguments.bab_branch_queue_priority_mode,
                self.arguments.torchonnx_model,
                self.arguments.num_samples_for_grad_estimation,
            )
            branches_counter[0] = branches_counter[0] + len(new_branches)
            self.branch_queue.put(new_branches)
