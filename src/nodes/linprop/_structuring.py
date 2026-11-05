__all__ = ["ConcatLinPropNode"]

from torch import Tensor

from src.nodes.basic import ConcatNode
from src.nodes.linprop._basic import BasicLinPropNode
from src.utils import store_and_update_bnd
from src.substorch import (
    backsub_concat,
    backsub_concat_lower,
    interval_concat,
)


class ConcatLinPropNode(ConcatNode, BasicLinPropNode):

    def forward(self):
        BasicLinPropNode.forward(self)

        pre_node_names = [pre_node.name for pre_node in self.pre_nodes]
        pre_node_dict = {node.name: node for node in self.pre_nodes}
        input_names = self.input_names
        pre_lower_bound_list, pre_upper_bound_list = [], []
        for name in input_names:
            if name in pre_node_names:
                pre_node = pre_node_dict[name]
                pre_bound = self._get_fwdprop_bnd(pre_node, name)
                pre_lower_bound_list.append(pre_bound[0])
                pre_upper_bound_list.append(pre_bound[1])
            else:
                constant_value = self.constants[name]
                pre_lower_bound_list.append(constant_value)
                pre_upper_bound_list.append(constant_value)

        new_bound = interval_concat(
            pre_lower_bound_list, pre_upper_bound_list, dim=self.dim
        )

        store_and_update_bnd(new_bound, self.cache, self)

    def _bwdprop_symbnd(self):

        dim = max(len(input_size) for input_size in self.input_sizes)
        new_input_sizes = []
        for input_size in self.input_sizes:
            if len(input_size) < dim:
                new_input_sizes.append((1,) + input_size)
            else:
                new_input_sizes.append(input_size)

        split_size_or_sections = [
            input_size[self.dim] for input_size in new_input_sizes
        ]

        symbnd = self._get_bwdprop_symbnd()
        if len(symbnd) == 4:
            (
                new_lower_symbnd_coeffs_list,
                new_lower_symbnd_const_list,
                new_upper_symbnd_coeffs_list,
                new_upper_symbnd_const_list,
            ) = backsub_concat(
                *symbnd, split_size_or_sections=split_size_or_sections, dim=self.dim
            )

            new_symbnd = tuple(
                (
                    new_lower_bound_coeffs,
                    new_lower_symbnd_const,
                    new_upper_symbnd_coeffs,
                    new_upper_symbnd_const,
                )
                for (
                    new_lower_bound_coeffs,
                    new_lower_symbnd_const,
                    new_upper_symbnd_coeffs,
                    new_upper_symbnd_const,
                ) in zip(
                    new_lower_symbnd_coeffs_list,
                    new_lower_symbnd_const_list,
                    new_upper_symbnd_coeffs_list,
                    new_upper_symbnd_const_list,
                )
            )
        else:
            assert len(symbnd) == 2
            (
                new_lower_symbnd_coeffs_list,
                new_lower_symbnd_const_list,
            ) = backsub_concat_lower(
                *symbnd, split_size_or_sections=split_size_or_sections, dim=self.dim
            )

            new_symbnd = tuple(
                (
                    new_lower_bound_coeffs,
                    new_lower_symbnd_const,
                )
                for new_lower_bound_coeffs, new_lower_symbnd_const in zip(
                    new_lower_symbnd_coeffs_list, new_lower_symbnd_const_list
                )
            )

        self.cache.symbnds[self.name] = new_symbnd
