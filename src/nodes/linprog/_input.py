__all__ = ["InputLinProgNode"]

from src.nodes.basic import InputNode
from src.nodes.linprog._basic import BasicLinProgNode


class InputLinProgNode(InputNode, BasicLinProgNode):

    def forward(self):
        BasicLinProgNode.forward(self)

    def add_constrs(self):
        pass
