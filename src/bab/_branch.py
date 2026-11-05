__all__ = ["Branch"]

from abc import ABC, abstractmethod

from torch import Tensor

from src.cache import BaBPropCache
from src.enums import *
from src.nodes import BasicNode


class Branch(ABC):
    _name: str
    _priority: float
    _or_groups: list[list[int]]
    _nodes: list[BasicNode]
    _cache: BaBPropCache | None
    name: str
    priority: float
    or_groups: list[list[int]]
    nodes: list[BasicNode]
    cache: BaBPropCache
    verbose: bool

    _each_prop_status: list[list[PropStatus]] | None
    each_prop_status: list[list[PropStatus]] | None
    _output_constrs: list[Tensor] | None
    output_constrs: list[Tensor] | None

    def __init__(
        self,
        name: str,
        priority: float,
        or_groups: list[list[int]],
        nodes: list[BasicNode],
        cache: BaBPropCache,
        *,
        verbose: bool = False,
    ):
        self._name = name
        self._priority = priority
        self._or_groups = or_groups
        self._nodes = nodes
        self._cache = cache
        self.verbose = verbose

        self._each_prop_status = [[PropStatus.UNKNOWN] * len(g) for g in or_groups]
        self._output_constrs = None

    @abstractmethod
    def split(self, *args, **kwargs) -> list["Branch"]:
        pass

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> float:
        return self._priority

    @property
    def or_groups(self) -> list[list[int]]:
        return self._or_groups

    @property
    def nodes(self) -> list[BasicNode]:
        return self._nodes

    @property
    def cache(self) -> BaBPropCache:
        return self._cache

    @property
    def each_prop_status(self) -> list[list[PropStatus]]:
        return self._each_prop_status

    @each_prop_status.setter
    def each_prop_status(self, each_prop_status: list[list[PropStatus]] | None):
        if each_prop_status is None:
            self._each_prop_status = None
            return
        assert all(
            len(each_prop_status[i]) == len(self._or_groups[i])
            for i in range(len(self._or_groups))
        )
        self._each_prop_status = each_prop_status

    def update_each_prop_status(self, value: list[list[PropStatus]]):
        assert all(
            len(value[i]) == len(self._or_groups[i])
            for i in range(len(self._or_groups))
        )
        for i in range(len(self._or_groups)):
            for j in range(len(self._or_groups[i])):
                new_value, old_value = value[i][j], self._each_prop_status[i][j]
                if new_value in (PropStatus.SAFE, PropStatus.UNSAFE):
                    assert old_value == PropStatus.UNKNOWN or old_value == new_value
                    self._each_prop_status[i][j] = new_value

    @property
    def output_constrs(self) -> list[Tensor] | None:
        return self._output_constrs

    def set_output_constrs(self, value: list[Tensor] | None):
        if value is None:
            self._output_constrs = None
            return

        assert all(len(c) == len(g) for c, g in zip(value, self._or_groups))
        assert all(c.size(1) == value[0].size(1) for c in value)
        self._output_constrs = value

    def __lt__(self, other) -> bool:
        return self.priority < other.priority

    def __le__(self, other) -> bool:
        return self.priority <= other.priority

    def __gt__(self, other) -> bool:
        return self.priority > other.priority

    def __ge__(self, other) -> bool:
        return self.priority >= other.priority

    def __eq__(self, other) -> bool:
        return self.priority == other.priority

    def __ne__(self, other) -> bool:
        return self.priority != other.prioritys

    def __str__(self):
        return f"{self.__class__.__name__} ({self._name})"

    def __repr__(self):
        return self.__str__()
