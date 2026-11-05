__all__ = ["BasicLinProgNode"]

from src.args import LinProgArgs
from src.cache import LinProgCache
from src.nodes.basic import BasicNode


class BasicLinProgNode(BasicNode):
    argument: LinProgArgs
    cache: LinProgCache

    def forward(self):
        self.add_vars()
        self.add_constrs()

    def add_vars(self):
        model = self.cache.lp_model
        lb, ub = self.cache.bnds[self.name]
        lb, ub = lb.flatten().tolist(), ub.flatten().tolist()
        n = len(lb)
        vars_list = []
        for i in range(n):
            var = model.add_var(name=f"{self.name}_{i}", lb=lb[i], ub=ub[i])
            vars_list.append(var)

        self.cache.lp_vars[self.name] = vars_list

    def add_constrs(self):
        raise RuntimeError(f"Node {self.name} does not implement add_constrs method.")
