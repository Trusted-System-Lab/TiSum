__all__ = ["RelaxMode", "ConstrTemplate"]

from enum import IntEnum, unique


@unique
class RelaxMode(IntEnum):
    """The activation relaxation mode."""

    SINGLE_LINEAR = 1
    """Single-Neuron Relaxation."""
    MULTI_LINEAR = 2
    """Multi-Neuron Relaxation."""

    CROWN = 3
    """CROWN Relaxation."""
    DEEPPOLY = 4
    """DeepPoly Relaxation."""
    DEEPZONO = 5
    """DeepZono Relaxation."""

    ADA_SINGLE_LINEAR = 6
    """Adaptive Single-Neuron Relaxation."""
    ADA_MULTI_LINEAR = 7
    """Adaptive Multi-Neuron Relaxation."""

    SINGLE_ZONO = 8
    """Single Zonotope Relaxation."""
    DOUBLE_ZONO = 9
    """Double Zonotope Relaxation."""

    SINGLE_QUAD = 10
    """Quadratic Relaxation."""

    def __repr__(self):
        return f"{self.name}"


@unique
class ConstrTemplate(IntEnum):
    """The constraint template used in the KACT multi-neuron calculation."""

    HEXAGON = 1
    """The hexagon constraint template."""
    RHOMBUS = 2
    """The rhombus constraint template."""

    def __repr__(self):
        return f"{self.name}"

    def __str__(self):
        return f"{self.name}"
