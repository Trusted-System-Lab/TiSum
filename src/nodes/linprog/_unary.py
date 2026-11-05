__all__ = ["ReLULinProgNode"]

from src.miplayer import add_constrs_relu
from src.args import LinProgArgs
from src.cache import LinProgCache
from src.nodes.basic import ReLUNode
from src.nodes.linprog._basic import BasicLinProgNode


class ReLULinProgNode(ReLUNode, BasicLinProgNode):
    argument: LinProgArgs
    cache: LinProgCache

    def forward(self):
        BasicLinProgNode.forward(self)

    def add_constrs(self):
        model = self.cache.lp_model
        x = self.cache.lp_vars[self.pre_nodes[0].name]
        y = self.cache.lp_vars[self.name]
        x_lb, x_ub = self.cache.bnds[self.pre_nodes[0].name]
        x_lb, x_ub = x_lb.flatten().cpu().numpy(), x_ub.flatten().cpu().numpy()
        max_num_mip_relu = (
            self.argument.lp_max_num_mip_relu - self.cache.lp_num_mip_relu
        )
        if not self.argument.use_mip:
            max_num_mip_relu = 0
        if max_num_mip_relu < 0:
            raise ValueError(
                f"Number of MIP ReLUs exceeds the limit: "
                f"{self.cache.lp_num_mip_relu} + {max_num_mip_relu} < 0"
            )
        constrs_list, num_mip_relu = add_constrs_relu(
            model,
            x,
            y,
            x_lb,
            x_ub,
            name=self.name,
            max_num_mip_relu=max_num_mip_relu,
            min_range_mip_relu=self.argument.lp_min_range_mip_relu,
        )

        self.cache.lp_num_mip_relu += num_mip_relu
        self.cache.lp_constrs[self.name] = constrs_list
