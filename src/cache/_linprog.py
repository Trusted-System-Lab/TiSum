__all__ = [
    "LinProgCache",
]

from dataclasses import dataclass, field

import mip
from mip import Var, Constr

from src.cache._boundprop import BasicCache


@dataclass(slots=True)
class LinProgCache(BasicCache):
    lp_model: mip.Model | None = None
    lp_vars: dict[str, list[Var]] = field(default_factory=dict)
    lp_constrs: dict[str, list[Constr]] = field(default_factory=dict)
    lp_num_mip_relu: int = 0

    def clear(
        self,
        keep_bnds: bool = False,
        keep_rlxs: bool = False,
        keep_symbnds: bool = False,
    ):
        BasicCache.clear(self, keep_bnds)
        self.lp_model = None
        self.lp_vars = {}
        self.lp_constrs = {}
        self.lp_num_mip_relu = 0



