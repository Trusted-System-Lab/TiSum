__all__ = ["PropStatus", "VerifStatus"]

from enum import IntEnum, unique


@unique
class PropStatus(IntEnum):
    """The status of a property."""

    SAFE = 1
    """The sub-problem is SAT, i.e., the property to verify is satisfied."""
    UNSAFE = 2
    """The sub-problem is UNSAT, i.e., the property to verify is not satisfied."""
    UNKNOWN = 3
    """The sub-problem is UNKNOWN, i.e., the property to verify is undecidable."""
    INFEASIBLE = 4
    """The sub-problem is INFEASIBLE, i.e., the property to verify is infeasible."""

    def __repr__(self):
        return f"{self.name}"

    def __str__(self):
        return f"{self.name}"


@unique
class VerifStatus(IntEnum):
    """The status of the verification result."""

    SAFE = 1
    """The property to verify is satisfied."""
    UNSAFE = 2
    """The property to verify is not satisfied."""
    UNKNOWN = 3
    """The property to verify is undecidable."""
    TIMEOUT = 4
    """The verification is timeout."""
    MEMOUT = 5
    """The verification is memoryout."""

    def __repr__(self):
        return f"{self.name}"

    def __str__(self):
        return f"{self.name}"
