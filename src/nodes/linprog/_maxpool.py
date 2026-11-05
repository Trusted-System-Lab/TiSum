__all__ = ["MaxPool2dLinProgNode"]

from src.miplayer import add_constrs_maxpool2d
from src.args import LinProgArgs
from src.cache import LinProgCache
from src.nodes.basic import MaxPool2dNode
from src.nodes.linprog._basic import BasicLinProgNode


class MaxPool2dLinProgNode(MaxPool2dNode, BasicLinProgNode):
    argument: LinProgArgs
    cache: LinProgCache
    input_sizes: tuple[tuple[int, int, int]]
    output_sizes: tuple[tuple[int, int, int]]
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]

    def forward(self):
        BasicLinProgNode.forward(self)

    def add_constrs(self):
        model = self.cache.lp_model
        x = self.cache.lp_vars[self.pre_nodes[0].name]
        y = self.cache.lp_vars[self.name]
        x_lb, x_ub = self.cache.bnds[self.pre_nodes[0].name]
        x_lb, x_ub = x_lb.flatten().numpy(), x_ub.flatten().numpy()
        constrs_list = add_constrs_maxpool2d(
            model,
            x,
            y,
            x_lb,
            x_ub,
            input_size=self.input_sizes[0],
            output_size=self.output_sizes[0],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            name=self.name,
            use_mip=self.argument.use_mip,
        )

        self.cache.lp_constrs[self.name] = constrs_list
