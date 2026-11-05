__all__ = ["BranchQueue"]

from queue import PriorityQueue

from src.bab._branch import Branch


class BranchQueue(PriorityQueue):
    """The branch queue for the branches."""

    def __init__(self, max_size: int = 0, max_batches: int = 2):
        super(BranchQueue, self).__init__(max_size)

        self.max_batches: int = max_batches
        self._current_size: int = 0
        self._total_size: int = 1

    def put(self, item: Branch | list[Branch], block=False, timeout=None):
        """Put the branch or branches into the queue."""
        if isinstance(item, list):
            for branch in item:
                super().put(branch, block, timeout)
                self._current_size += 1
                self._total_size += 1

        elif isinstance(item, Branch):
            super().put(item, block, timeout)
            self._current_size += 1
            self._total_size += 1

            parent_name = item.name[:-1]
            if len(parent_name) == 0:
                return

        else:
            raise ValueError(f"Unsupported type {type(item)} for put().")

    def get_next_batch(self) -> list[Branch]:
        branches = []
        n_branch = 0
        while n_branch < self.max_batches:
            if self.empty():
                break

            branch: Branch = self.get()
            branches.append(branch)
            n_branch += 1
            self._current_size -= 1

        return branches

    @property
    def current_size(self):
        return self._current_size

    @property
    def total_size(self):
        return self._total_size

    def __str__(self):
        return (
            f"{self.__class__.__name__} (size={self.qsize()}, maxsize={self.maxsize})"
        )

    def __repr__(self):
        return self.__str__()
