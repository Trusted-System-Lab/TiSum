__all__ = ["BatchedQuadPropModel"]

from src.args import BatchedQuadPropArgs
from src.batched_model.linprop import BatchedLinPropModel
from src.cache import QuadPropCache
from src.model import QuadPropModel


class BatchedQuadPropModel(BatchedLinPropModel, QuadPropModel):
    arguments: BatchedQuadPropArgs
    cache: QuadPropCache

    pass
