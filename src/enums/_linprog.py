__all__ = ["OptMethod"]

from enum import IntEnum, unique


@unique
class OptMethod(IntEnum):
    """The optimization method after the bound propagation method."""

    LP = 1
    """Construct linear programming with single-neuron constraints."""
    KACT = 2
    """
    Construct linear programming with multi-neuron constraints by small neuron group 
    strategy.
    """

    def __repr__(self):
        return f"{self.name}"
