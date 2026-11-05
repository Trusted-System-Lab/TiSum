__all__ = ["BatchedAdaPropModel"]

import torch
from torch import Tensor

from src.batched_model.linprop import BatchedLinPropModel
from src.cache import AdaPropCache
from src.model import AdaPropModel
from src.nodes import BatchedBasicLinPropNode, BasicLinPropNode


class BatchedAdaPropModel(BatchedLinPropModel, AdaPropModel):
    cache: AdaPropCache

    def _cal_ada_loss(self, is_lower: bool = True) -> Tensor:
        bound = self.cache.bnds[self.cache.cur_node.name]
        if is_lower:
            return -torch.mean(bound[0])
        return torch.mean(bound[1])

    def _stop_optimization(self, node: BatchedBasicLinPropNode):
        return False

    def backsub(self, node: BasicLinPropNode):
        return AdaPropModel.backsub(self, node)
