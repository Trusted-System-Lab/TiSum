__all__ = ["QuadPropModel"]

from src.args import QuadPropArgs
from src.cache import QuadPropCache
from src.model.linprop import LinPropModel


class QuadPropModel(LinPropModel):
    arguments: QuadPropArgs
    cache: QuadPropCache

    pass
