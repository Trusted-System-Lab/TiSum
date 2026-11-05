__all__ = ["PropMode"]

from enum import IntEnum, unique


@unique
class PropMode(IntEnum):
    """Enumeration of propagation modes."""

    FORWARD = 1
    BACKWARD = 2
