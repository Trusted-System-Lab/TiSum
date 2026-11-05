__all__ = ["LinProgModel"]

import time
import warnings

import mip
import numpy as np
import torch
from mip import OptimizationStatus
from torch import Tensor

from src.args import LinProgArgs
from src.cache import LinProgCache
from src.enums import PropStatus
from src.model.advattack import AdvAttackModel
from src.model.basic import BasicModel
from src.model.ort_session_infer import run_onnx_model
from src.nodes import BasicLinProgNode


class LinProgModel(BasicModel):
    _lp_model: mip.Model | None = None
    _adv_attack_model: AdvAttackModel | None = None
    arguments: LinProgArgs
    cache: LinProgCache

    def __init__(
        self,
        nodes: list[BasicLinProgNode],
        arguments: LinProgArgs,
        verbose: bool = False,
    ):
        BasicModel.__init__(self, nodes, arguments, verbose)

        if self.verbose:
            print(f"Set LP solver as {arguments.lp_solver}.")

        self._lp_model = None
        self._adv_attack_model = None

    def set_model_params(self):
        self._lp_model.verbose = False
        self._lp_model.seed = self.arguments.random_seed
        self._lp_model.max_seconds = self.arguments.lp_timelimit
        self._lp_model.cutoff = self.arguments.lp_lower_cutoff

        if self.arguments.lp_solver == mip.GRB:

            solver: mip.gurobi.SolverGurobi = self._lp_model.solver
            solver.set_verbose(0)
            solver.set_int_param("OutputFlag", 0)
            solver.set_int_param("LogToConsole", 0)
            solver.set_int_param("Method", self.arguments.grb_method)
            solver.set_int_param("NumericFocus", self.arguments.grb_numeric_focus)
            solver.set_dbl_param("OptimalityTol", self.arguments.grb_optimality_tol)

            solver.set_int_param("MIPFocus", 1)
            solver.set_int_param("Cuts", 1)
            solver.set_int_param("VarBranch", 1)
            solver.set_dbl_param("Heuristics", 0.5)

    def add_vars_constrs(self):
        self.cache.lp_num_mip_relu = 0

        node = self.nodes[0]
        node.forward()

        for i in range(1, len(self.nodes)):
            node = self.nodes[i]
            node.forward()

        if self.verbose:
            print(
                f"Build LP model with "
                f"{len(self._lp_model.vars)} vars, "
                f"{self.cache.lp_num_mip_relu} MIP ReLU vars, and "
                f"{len(self._lp_model.constrs)} constraints."
            )

    def optimize_lp(
        self,
        input_constrs: Tensor,
        output_constrs: list[Tensor],
        output_weight: Tensor,
        output_bias: Tensor,
    ):
        self.lp_model.sense = mip.MINIMIZE
        input_vars = self.cache.lp_vars[self.nodes[0].name]
        output_vars = self.cache.lp_vars[self.nodes[-1].name]

        self._each_prop_status = [
            [PropStatus.UNKNOWN for _ in or_group] for or_group in self._or_groups
        ]

        output_bound = self.cache.bnds[self.nodes[-1].name]

        obj_val = None
        for i, or_group in enumerate(self._or_groups):
            start, end = or_group[0], or_group[-1] + 1
            lower_bound = output_bound[0][start:end]
            upper_bound = output_bound[1][start:end]

            status = PropStatus.UNKNOWN
            is_single_prop = len(or_group) == 1
            for j, (lower, upper) in enumerate(zip(lower_bound, upper_bound)):

                if lower.item() > 0:
                    self._each_prop_status[i][j] = PropStatus.SAFE
                    break



                status, obj_val, adv_example = self._optimize_one_output_var(
                    input_vars,
                    output_vars[start + j],
                    start + j,
                    input_constrs=input_constrs,
                    output_constrs=output_constrs,
                    output_weight=output_weight,
                    output_bias=output_bias,
                    is_single_prop=is_single_prop,
                    is_lower=True,
                    is_mip=self.arguments.use_mip,
                )
                if status == PropStatus.UNKNOWN and obj_val is None:
                    break
                if obj_val is not None and obj_val > lower_bound[j]:
                    lower_bound[j] = obj_val
                if status == PropStatus.SAFE:
                    self._each_prop_status[i][j] = PropStatus.SAFE
                    break
                if adv_example is not None:
                    for k in range(len(self._each_prop_status[i])):
                        self._each_prop_status[i][k] = PropStatus.UNSAFE
                    break
                if status == PropStatus.UNSAFE:
                    self._each_prop_status[i][j] = PropStatus.UNSAFE
                    self._adv_example = adv_example
                    continue



            output_bound[0][start:end] = lower_bound
            output_bound[1][start:end] = upper_bound
            if all(s == PropStatus.UNSAFE for s in self._each_prop_status[i]):
                break

            if status == PropStatus.UNKNOWN and obj_val is None:
                break

        self.cache.bnds[self.nodes[-1].name] = output_bound

    def find_adv_example(
        self,
        input_constrs: Tensor,
        output_constrs: list[Tensor],
    ) -> tuple[Tensor, Tensor] | None:
        if self.adv_attack_model is None:
            raise ValueError("Adv attack model is not set for LinProgModel.")
        if self.verbose:
            print("Run adv attack model to find adversarial example.")
        attack_model = self.adv_attack_model
        attack_model.run(input_constrs, output_constrs)
        return attack_model.adv_example

    def _optimize_one_output_var(
        self,
        input_vars: list[mip.Var],
        output_var: mip.Var,
        output_idx: int,
        *,
        input_constrs: Tensor,
        output_constrs: list[Tensor],
        output_weight: Tensor,
        output_bias: Tensor,
        is_single_prop: bool,
        is_lower: bool,
        is_mip: bool = False,
        use_cutoff: bool = True,
    ) -> tuple[PropStatus, float | None, tuple[Tensor, Tensor] | None]:
        model = self.lp_model
        model.objective = mip.xsum([output_var])

        args = self.arguments

        model.sense = mip.MINIMIZE
        solver: mip.gurobi.SolverGurobi = model.solver
        if is_lower:
            model.sense = mip.MINIMIZE
            if use_cutoff:
                model.cutoff = args.lp_lower_cutoff
                solver.set_dbl_param("BestBdStop", args.lp_lower_cutoff)
            else:
                model.cutoff = float("inf")
                solver.set_dbl_param("BestBdStop", float("inf"))
        else:
            model.sense = mip.MAXIMIZE
            if use_cutoff:
                model.cutoff = args.lp_upper_cutoff
                solver.set_dbl_param("BestBdStop", args.lp_upper_cutoff)
            else:
                model.cutoff = float("-inf")
                solver.set_dbl_param("BestBdStop", float("-inf"))

        model.optimize(args.lp_timelimit)

        status = model.status
        bound_name = "lower bound" if is_lower else "upper bound"
        obj_val = output_var.x
        if obj_val is not None:
            obj_val = float(obj_val)

        def get_adv_example() -> tuple[Tensor, Tensor] | None:
            nonlocal input_vars
            adv_x = [var.x for var in input_vars]
            adv_x = np.asarray(adv_x, dtype=np.float32)
            adv_y = run_onnx_model(args.onnx_model, adv_x)
            dtype, device = self.cache.dtype, self.cache.device
            adv_x = torch.tensor(adv_x, dtype=dtype, device=device)
            adv_y = torch.tensor(adv_y, dtype=dtype, device=device)
            adv_z = adv_y @ output_weight.t() + output_bias

            if torch.any(adv_z.flatten() > 0.0):
                return None
            return adv_x, adv_y

        if status == OptimizationStatus.OPTIMAL:
            if self.verbose:
                print(
                    f"Optimal {bound_name} of {output_var.name} = {obj_val:.4f}, "
                    f"use_mip={is_mip}"
                )

            if obj_val > 0.0:
                if is_lower:
                    return PropStatus.SAFE, obj_val, None
                return PropStatus.UNKNOWN, obj_val, None

            if (
                is_lower
                and is_mip
                and self.cache.lp_num_mip_relu < self.arguments.lp_max_num_mip_relu
            ) or not is_lower:
                adv_example = get_adv_example()
                return PropStatus.UNSAFE, obj_val, adv_example
            return PropStatus.UNKNOWN, obj_val, None

        elif status == OptimizationStatus.CUTOFF:
            obj_val = args.lp_lower_cutoff if is_lower else args.lp_upper_cutoff
            if self.verbose:
                print(f"Cutoff {bound_name} of {output_var.name} >= {obj_val}")

            if is_lower:
                return PropStatus.SAFE, obj_val, None
            return PropStatus.UNSAFE, obj_val, None

        elif status == OptimizationStatus.FEASIBLE:
            objective_bound = model.objective_bound
            if self.verbose:
                print(
                    f"Feasible {bound_name} of {output_var.name} = {obj_val:.4f}, "
                    f"objective bound: {objective_bound:.4f}, "
                )
            if is_lower:
                if float(objective_bound) > 0.0:
                    return PropStatus.SAFE, objective_bound, None
                return PropStatus.UNKNOWN, objective_bound, None
            if objective_bound <= 0.0:
                return PropStatus.UNSAFE, objective_bound, None
            return PropStatus.UNKNOWN, objective_bound, None

        elif status == OptimizationStatus.ERROR:
            if self.verbose:
                print(f"Error in optimizing {bound_name} of {output_var.name}.")

            warnings.warn(
                f"Error in optimizing {bound_name} of {output_var.name}. "
                f"Status: {status.name}"
            )
            return PropStatus.UNKNOWN, None, None

        elif status in {
            OptimizationStatus.INFEASIBLE,
            OptimizationStatus.INT_INFEASIBLE,
        }:
            if self.verbose:
                print(f"Infeasible {bound_name} of {output_var.name}.")

            warnings.warn(
                f"This should not happen. Infeasible {bound_name} of {output_var.name}. "
                f"Status: {status.name}"
            )
            return PropStatus.UNKNOWN, None, None

        elif status == OptimizationStatus.UNBOUNDED:
            if self.verbose:
                print(f"Unbounded {bound_name} of {output_var.name}.")

            if use_cutoff:
                return self._optimize_one_output_var(
                    input_vars,
                    output_var,
                    output_idx,
                    input_constrs=input_constrs,
                    output_constrs=output_constrs,
                    output_weight=output_weight,
                    output_bias=output_bias,
                    is_single_prop=is_single_prop,
                    is_lower=is_lower,
                    is_mip=is_mip,
                    use_cutoff=False,
                )


            for v in model.vars:
                if self.verbose:
                    print("Checking variable:", v.name)
                model.objective = mip.xsum([v])
                for sense in {mip.MINIMIZE, mip.MAXIMIZE}:
                    model.sense = sense
                    model.cutoff = (
                        float("inf") if sense == mip.MINIMIZE else float("-inf")
                    )
                    model.optimize(self.arguments.lp_timelimit)
                    if self.verbose:
                        print(f"Variable {v.name} bound: {v.x}")
                    if model.status == OptimizationStatus.UNBOUNDED:
                        if self.verbose:
                            print(f"Find unbounded variable: {v.name}")

            warnings.warn(
                f"This should not happen. Unbounded {bound_name} of {output_var.name}. "
                f"Status: {status.name}"
            )
            return PropStatus.UNKNOWN, None, None

        if self.verbose:
            print(
                f"Status of {bound_name} of {output_var.name} is {status}. "
                f"Set to UNKNOWN."
            )
        return PropStatus.UNKNOWN, None, None

    def run(
        self,
        input_constrs: Tensor,
        output_constrs: list[Tensor],
    ):
        self._adv_example = None
        self._lp_model = mip.Model(solver_name=self.arguments.lp_solver)
        self.cache.lp_model = self._lp_model
        self.set_model_params()

        self.set_input_constrs(input_constrs)
        output_weight, output_bias = self.set_output_constrs(output_constrs)

        self.add_vars_constrs()

        t = time.perf_counter()
        self.optimize_lp(input_constrs, output_constrs, output_weight, output_bias)
        time_solve = time.perf_counter() - t
        if self.verbose:
            print(f"LP optimization time: {time_solve:.2f}s.")

    @property
    def lp_model(self) -> mip.Model:
        return self._lp_model

    def set_adv_attack_model(self, adv_attack_model: AdvAttackModel):
        self._adv_attack_model = adv_attack_model

    @property
    def adv_attack_model(self) -> AdvAttackModel:
        return self._adv_attack_model
