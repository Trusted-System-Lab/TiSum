__all__ = ["BaBMode", "BranchPriorMode", "SplitMode"]

from enum import IntEnum, unique


@unique
class BaBMode(IntEnum):
    """The mode used in the branch and bound algorithm."""

    SPLIT_INPUT = 1
    """The input splitting mode."""
    SPLIT_NEURON = 2
    """The ReLU splitting mode, i.e, split the ReLU activation at x=0."""
    BATCHED_SPLIT_INPUT = 3
    """The batched input splitting mode."""

    def __repr__(self):
        return f"{self.name}"


@unique
class BranchPriorMode(IntEnum):
    """The priority mode used in the branch and bound algorithm."""

    FSB = 1
    """
    The priority is based on the FSB (Filtered Smart Branching) strategy.

    .. seealso::
        This implementation is based on the following papers but with some 
        modifications to adjust the bound propagation method.

        - `Branch and Bound for Piecewise Linear Neural Network Verification
          <https://arxiv.org/pdf/1909.06588>`__
          :cite:`bunel_branch_2020`
        - `Improved Branch and Bound for Neural Network Verification via Lagrangian
          Decomposition
          <https://arxiv.org/pdf/2104.06718>`__
          :cite:`de_palma_improved_2021`
    """
    BBPS = 2
    """
    The priority is based on the BBPS (Bound Propagation with Shortcuts) strategy.

    .. seealso::

        - (GenBaB) `Formal Verification for Neural Networks with General Nonlinearities 
          via Branch-and-bound
          <https://arxiv.org/pdf/2405.21063>`__
          :cite:`shi_formal_2023`
    """

    MIN_DISTANCE_ZERO = 3
    """
    This is to minimize the distance to zero. This is like DFS (Depth First Search). 
    This will keep as small as possible cached branches.
    """
    MAX_DISTANCE_ZERO = 4
    """
    This is to maximize the distance to zero. This is like BFS (Breadth First Search).
    This will keep as small as possible the global bound of the branches.
    """
    MIN_LOWER_BOUND = 5
    """
    This is to minimize the lower bound of the branches.
    This is like best-first search.
    """
    MAX_LOWER_BOUND = 6
    """
    This is to maximize the lower bound of the branches.
    This is like worst-first search.
    """

    def __repr__(self):
        return f"{self.name}"


@unique
class SplitMode(IntEnum):
    BISECTION = 1
    EFFECT = 2
    GRAD = 3
