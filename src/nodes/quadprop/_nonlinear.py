__all__ = ["NonlinearQuadPropNode"]

from src.cache import QuadPropCache
from src.utils import store_and_update_bnd
from src.substorch import (
    cal_scalar_bound,
    cal_scalar_bound_lower,
    cal_quad_scalar_bound,
    cal_quad_scalar_bound_lower,
)


class NonlinearQuadPropNode:
    """This is an interface for unary activation function propagation nodes."""

    name: str
    cache: QuadPropCache

    def _cal_and_update_cur_node_bnd(self):
        """This method calculate the scalar bounds based on the input bounds."""
        symbnd = self.cache.symbnds[self.name]
        bound = self.cache.bnds[self.name]
        if len(symbnd) in {2, 4}:
            if len(symbnd) == 2:
                new_bound = cal_scalar_bound_lower(*symbnd, *bound)
            else:
                new_bound = cal_scalar_bound(*symbnd, *bound)
        elif len(symbnd) in {3, 6}:
            if len(symbnd) == 3:
                new_bound = cal_quad_scalar_bound_lower(*symbnd, *bound)
            else:
                new_bound = cal_quad_scalar_bound(*symbnd, *bound)
        else:
            raise ValueError(
                f"Invalid symbolic bound length: {len(symbnd)} for node {self.name}."
            )

        store_and_update_bnd(new_bound, self.cache, self.cache.cur_node)
