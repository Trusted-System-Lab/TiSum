__all__ = ["AdaPropModel"]

import time

import torch
from torch import Tensor

from src.args import AdaPropArgs
from src.cache import AdaPropCache
from src.enums import PropStatus
from src.model.linprop import LinPropModel
from src.model.utils import analyze_status, analyze_each_prop_status
from src.nodes import BasicLinPropNode, OutputNode
from src.utils.loss import *
from src.substorch import pgd_nd_duchi


class AdaPropModel(LinPropModel):
    arguments: AdaPropArgs
    cache: AdaPropCache

    def _collect_rlx_params(self, node: BasicLinPropNode) -> list[Tensor]:
        params = []
        backward_sort = self._all_backward_sorts[node]
        i = 0
        for j in range(1, len(backward_sort)):
            node = backward_sort[j]
            ada_param = self.cache.ada_rlx_prms.get(node.name)
            if ada_param is None:
                continue

            lower_param, upper_param = ada_param
            if lower_param is not None:
                assert (
                    lower_param.requires_grad
                ), "Lower relaxation parameter must require gradient."
                params.append(lower_param)
            if upper_param is not None:
                assert (
                    upper_param.requires_grad
                ), "Upper relaxation parameter must require gradient."
                params.append(upper_param)

            i += 1

        if self.verbose:
            print(f"Collect {len(params)} parameters for node {node.name}.")

        return params

    @staticmethod
    def _project_rlx_params(params: list[Tensor]):
        for param in params:
            if param.dim() == 1:
                param.data.clamp_(0, 1)
            else:
                param.data = pgd_nd_duchi(param.data)

    def _set_ada_optimizer(self, params: list[Tensor]):
        lr = self.arguments.ada_lr
        weight_decay = self.arguments.ada_weight_decay
        if self.arguments.ada_optimizer == "adam":
            from torch.optim import Adam

            optimizer = Adam(params, lr=lr, weight_decay=weight_decay)
        else:
            raise NotImplementedError(
                f"Optimizer {self.arguments.ada_optimizer} is not implemented."
            )

        return optimizer

    def _cal_ada_loss(self, is_lower: bool = True) -> Tensor:
        bound = self.cache.bnds[self.cache.cur_node.name]
        lower, upper = bound

        if not isinstance(self.cache.cur_node, OutputNode):
            if is_lower:
                return -torch.mean(lower)
            return torch.mean(upper)

        if is_lower:
            lower = lower.unsqueeze(0)
            return loss_maximize_min_or_group(lower, self.or_groups)
        upper = upper.unsqueeze(0)
        return loss_minimize_max_or_group(upper, self.or_groups)

    def _stop_optimization(self, node: BasicLinPropNode) -> bool:
        if not isinstance(node, OutputNode):
            return False

        output_bound = self.cache.bnds[node.name]
        each_prop_status = analyze_each_prop_status(output_bound, self.or_groups)
        status = analyze_status(each_prop_status)
        if status in {PropStatus.SAFE, PropStatus.UNSAFE}:
            return True
        return False

    def backsub(self, node: BasicLinPropNode):
        if self._without_backsub(node):
            return

        if self.arguments.ada_only_last_layer:
            if not isinstance(node, OutputNode):
                if self.verbose:
                    print(f"Skip optimization for {node.name}.")
                self._backsub(node)
                return

        params = self._collect_rlx_params(node)
        if not params:
            if self.verbose:
                print(f"No parameters to optimize for {node.name}.")
            self._backsub(node)
            return

        optimizer = self._set_ada_optimizer(params)

        def _optimize(is_lower: bool):
            time_start = time.perf_counter()

            last_loss = float("inf")
            epoch = 0
            for epoch in range(self.arguments.ada_max_iters):
                self.cache.detach()
                optimizer.zero_grad()

                if node.name not in self.cache.symbnds:
                    node.forward()
                self._backsub(node)

                loss = self._cal_ada_loss(is_lower)
                loss.backward()
                optimizer.step()

                self._project_rlx_params(params)
                if self._stop_optimization(node):
                    break

                cur_loss = loss.item()
                if abs(last_loss - cur_loss) < self.arguments.ada_min_loss_change:
                    break

                last_loss = cur_loss

            if self.verbose:
                print(
                    f"Optimization for {'lower' if is_lower else 'upper'} bounds "
                    f"of {node.name} completed in "
                    f"{time.perf_counter() - time_start:.2f}s "
                    f"after {epoch + 1} epochs."
                )

        if self.verbose:
            print(f"Optimizing lower bounds for {node.name}.")
        _optimize(is_lower=True)

        if self.verbose:
            print(f"Optimizing upper bounds for {node.name}.")
        _optimize(is_lower=False)

        node.forward()
        self._backsub(node)

        self.cache.detach()
        self.cache.cur_node = None

