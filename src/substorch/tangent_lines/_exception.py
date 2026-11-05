__all__ = ["NotConverged"]


class NotConverged(Exception):
    """An exception for not converged calculation in an algorithm."""

    def __init__(self, message="The calculation is not converged."):
        super().__init__(message)

    def __str__(self):
        return f"{self.__class__.__name__}: {super().__str__()}"
