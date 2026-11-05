__all__ = ["Conv2dLinProgNode"]

from src.miplayer import add_constrs_conv2d
from src.args import LinProgArgs
from src.cache import LinProgCache
from src.nodes.basic import Conv2dNode
from src.nodes.linprog._basic import BasicLinProgNode


class Conv2dLinProgNode(Conv2dNode, BasicLinProgNode):
    argument: LinProgArgs
    cache: LinProgCache
    input_sizes: tuple[tuple[int, int, int]]
    output_sizes: tuple[tuple[int, int, int]]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]

    def forward(self):
        BasicLinProgNode.forward(self)

    def add_constrs(self):
        model = self.cache.lp_model
        x = self.cache.lp_vars[self.pre_nodes[0].name]
        y = self.cache.lp_vars[self.name]
        weight = self.weight.cpu().numpy()
        bias = self.bias.cpu().numpy()
        constrs_list = add_constrs_conv2d(
            model,
            x,
            y,
            weight,
            bias,
            input_size=self.input_sizes[0],
            output_size=self.output_sizes[0],
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            name=self.name,
        )

        self.cache.lp_constrs[self.name] = constrs_list
