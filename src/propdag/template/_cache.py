__all__ = ["TCache"]

from abc import ABC
from dataclasses import dataclass


@dataclass(slots=True)
class TCache(ABC):
    """Abstract base class for caching in computational graphs."""

    pass
