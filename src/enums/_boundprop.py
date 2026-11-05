__all__ = ["BoundPropMode", "AdaActRelaxOptObj"]

from enum import IntEnum, unique


@unique
class BoundPropMode(IntEnum):
    """The bound propagation method."""

    INEQ = 1
    """Taking inequalities to over-approximate the range of each neuron."""
    ZONO = 2
    """Taking zonotopes to over-approximate the range of each neuron."""

    def __repr__(self):
        return f"{self.name}"


class AdaActRelaxOptObj(IntEnum):
    """The objective function used in the optimization for adaptive relaxation."""

    RANGE = 1
    """To minimize the sum of output ranges of neurons."""
    LOWER_BOUND = 2
    """To minimize the sum of lower bounds of neurons."""
    UPPER_BOUND = 3
    """To minimize the sum of upper bounds of neurons"""

    def __repr__(self):
        return f"{self.name}"
