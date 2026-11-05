__all__ = ["BatchedNonlinearQuadPropNode"]

from src.cache import QuadPropCache
from src.utils import batched_store_and_update_bnd
from src.substorch import (
    batched_cal_scalar_bound,
    batched_cal_scalar_bound_lower,
    batched_cal_quad_scalar_bound,
    batched_cal_quad_scalar_bound_lower,
)


class BatchedNonlinearQuadPropNode:
    """This is an interface for unary activation function propagation nodes."""

    name: str
    cache: QuadPropCache

    def _cal_and_update_cur_node_bnd(self):
        """This method calculate the scalar bounds based on the input bounds."""
        symbnd = self.cache.symbnds[self.name]
        bound = self.cache.bnds[self.name]
        if len(symbnd) in {3, 5}:
            if len(symbnd) == 3:
                new_bound = batched_cal_scalar_bound_lower(*symbnd, *bound)
            else:
                new_bound = batched_cal_scalar_bound(*symbnd, *bound)
        elif len(symbnd) in {4, 6}:
            if len(symbnd) == 4:
                new_bound = batched_cal_quad_scalar_bound_lower(*symbnd, *bound)
            else:
                new_bound = batched_cal_quad_scalar_bound(*symbnd, *bound)
        else:
            raise ValueError(
                f"Invalid symbolic bound length: {len(symbnd)} for node {self.name}."
            )

        batched_store_and_update_bnd(new_bound, self.cache, self.cache.cur_node)
