__all__ = [
    "SplitInputDict",
    "clamp_intput_bounds_by_lower_output_final_symbnds",
    "split_input_by_bisection",
    "split_input_by_effect",
    "split_input_by_grad",
    "cal_priority_by_min_lower_bound",
    "cal_priority_by_max_lower_bound",
    "cal_priority_by_min_distance_zero",
    "cal_priority_by_max_distance_zero",
]

from copy import deepcopy

import torch
from torch import Tensor

from src.model.utils import split_coeffs_and_const


class SplitInputDict(dict[int, tuple[float, float]]):
    """This is a dictionary of the split input tuple, where - the key is the input index and - the value is the range tuple of two floats, where - the first float is the lower bound and - the second float is the upper bound."""

    pass


def clamp_intput_bounds_by_lower_output_final_symbnds(
    input_bound: tuple[Tensor, Tensor],
    output_final_symbnds: tuple[Tensor, Tensor, Tensor, Tensor],
    or_groups: list[list[int]],
    verbose: bool = False,
) -> tuple[Tensor, Tensor]:
    """Clamp the input bounds by considering those input domain that contains the non-positive output symbolic bounds."""
    if sum(len(g) for g in or_groups) != 1:
        return input_bound

    l, u = input_bound
    xl, xu = l.flatten(), u.flatten()
    r = xu - xl

    w = output_final_symbnds[0].flatten(1)
    b = output_final_symbnds[1]
    wp, wn = w.clamp_min(0.0), w.clamp_max(0.0)

    wp_xl, wn_xu = wp @ xl, wn @ xu
    y = wp_xl + wn_xu + b
    y_u = wp @ xu + wn @ xl + b

    if torch.all(y_u <= 0.0):
        return input_bound

    incre = torch.abs(w * r)
    y_after = y.unsqueeze(1) + incre

    y_over = torch.clamp_min(y_after, 0.0)
    mask_pos = y_over > 0.0
    if not torch.any(mask_pos):
        return input_bound

    x_over = torch.zeros_like(y_over)
    mask = mask_pos & (w != 0.0)
    x_over[mask] = y_over[mask] / w[mask]
    x_over_p = x_over.clamp_min(0.0)
    x_over_n = x_over.clamp_max(0.0)

    x_over_p_final = torch.min(x_over_p, dim=0).values
    x_over_n_final = torch.max(x_over_n, dim=0).values
    xu_new = xu - x_over_p_final
    xl_new = xl - x_over_n_final

    try:
        assert torch.all(xl_new <= xu_new)
        assert torch.all(xl_new >= xl)
        assert torch.all(xu_new <= xu)
    except AssertionError:
        if verbose:
            print("input_bound[0]:\n", input_bound[0])
            print("input_bound[1]:\n", input_bound[1])
            print("output_final_symbnds[0]:\n", output_final_symbnds[0])
            print("output_final_symbnds[1]:\n", output_final_symbnds[1])
            print("or_groups:\n", or_groups)
            print("xl:\n", xl)
            print("xu:\n", xu)
            print("xl_new:\n", xl_new)
            print("xu_new:\n", xu_new)
        raise AssertionError


    l = xl_new.reshape_as(l).clone().detach()
    u = xu_new.reshape_as(u).clone().detach()

    return l, u


def split_input_by_bisection(
    input_bound: tuple[Tensor, Tensor],
    split_inputs: SplitInputDict,
    verbose: bool = False,
) -> list[SplitInputDict]:
    l, u = input_bound
    l, u = l.flatten(), u.flatten()
    max_range, idx_max_range = torch.max(u - l, dim=0)
    max_range, idx_max_range = max_range.item(), idx_max_range.item()

    l_selected = l[idx_max_range].item()
    u_selected = u[idx_max_range].item()
    m = (l_selected + u_selected) * 0.5

    split_inputs1 = deepcopy(split_inputs)
    split_inputs2 = deepcopy(split_inputs)
    split_inputs1[idx_max_range] = (l_selected, m)
    split_inputs2[idx_max_range] = (m, u_selected)


    return [split_inputs1, split_inputs2]


def split_input_by_effect(
    input_bound: tuple[Tensor, Tensor],
    split_inputs: SplitInputDict,
    output_final_symbnds: tuple[Tensor, Tensor, Tensor, Tensor],
    verbose: bool = False,
) -> list[SplitInputDict]:

    l, u = input_bound
    x_l = l.flatten()
    x_u = u.flatten()
    x_range = x_u - x_l

    w = output_final_symbnds[0]
    b = output_final_symbnds[1]
    w = w.flatten(1)
    w_p = w.clamp_min(0.0)
    w_n = w.clamp_max(0.0)

    y_l = w_p @ x_l + w_n @ x_u + b
    y_u = w_p @ x_u + w_n @ x_l + b
    y_r = y_u - y_l

    indicator = None
    x_offsets = None
    use_zero_point = False
    mask_batch = (y_l < 0.0) & (y_u > 0.0)
    if torch.any(mask_batch):
        y_range_masked = y_r * mask_batch.float()
        idx_max_range = torch.argmax(y_range_masked).item()
        w_selected = w[idx_max_range]
        y_l_selected = y_l[idx_max_range]
        dist_zero = -y_l_selected.item()

        x_offsets = dist_zero / w_selected.abs()
        x_offsets[x_offsets == float("inf")] = 0.0
        mask_valid = (x_offsets < x_range) & (x_offsets > 0.0)
        if torch.any(mask_valid):
            use_zero_point = True
            indicator = mask_valid.float() * x_offsets * (x_range - x_offsets)

    if not use_zero_point:
        y_idx_max_range = torch.argmax(y_r)
        indicator = w[y_idx_max_range].abs() * x_range


    selected_idx_max = torch.argmax(indicator).item()
    l_selected = x_l[selected_idx_max].item()
    u_selected = x_u[selected_idx_max].item()

    if use_zero_point:
        offset_selected = x_offsets[selected_idx_max].item()
        m = l_selected + offset_selected
    else:
        m = (l_selected + u_selected) * 0.5

    split_inputs1 = deepcopy(split_inputs)
    split_inputs2 = deepcopy(split_inputs)
    split_inputs1[selected_idx_max] = (l_selected, m)
    split_inputs2[selected_idx_max] = (m, u_selected)
    assert l_selected <= m <= u_selected


    return [split_inputs1, split_inputs2]


def split_input_by_grad(
    input_bound: tuple[Tensor, Tensor],
    split_inputs: SplitInputDict,
    model: torch.nn.Module,
    output_constrs: list[Tensor],
    num_samples: int = 64,
    verbose: bool = False,
) -> list[SplitInputDict]:
    weight, bias = split_coeffs_and_const(output_constrs)

    l, u = input_bound
    l, u = l.flatten(), u.flatten()
    r = u - l
    dtype, device = l.dtype, l.device

    x = torch.rand((num_samples, len(l)), dtype=dtype, device=device) * r + l
    x.detach_().requires_grad_()
    model.eval()

    y = model(x)
    z = y @ weight.t() + bias
    z = z.sum()
    grads = torch.autograd.grad(z, x, retain_graph=False, create_graph=False)[0]

    grad_range = grads.max(dim=0).values - grads.min(dim=0).values
    indicator = grad_range * r
    split_dim = torch.argmax(indicator).item()
    x_vals = x[:, split_dim].detach()
    grad_vals = grads[:, split_dim].detach()
    sorted_indices = torch.argsort(x_vals)
    sorted_x = x_vals[sorted_indices]
    sorted_grad = grad_vals[sorted_indices]
    grad_diff = torch.abs(sorted_grad[1:] - sorted_grad[:-1])
    max_diff_idx = torch.argmax(grad_diff).item()
    split_point = 0.5 * (sorted_x[max_diff_idx] + sorted_x[max_diff_idx + 1]).item()

    l_selected = l[split_dim].item()
    u_selected = u[split_dim].item()

    m = split_point
    m2 = (l_selected + u_selected) * 0.5
    m = (m + m2) * 0.5

    split_inputs1 = deepcopy(split_inputs)
    split_inputs2 = deepcopy(split_inputs)
    split_inputs1[split_dim] = (l_selected, m)
    split_inputs2[split_dim] = (m, u_selected)


    return [split_inputs1, split_inputs2]


def cal_priority_by_min_lower_bound(
    output_bound: tuple[Tensor, Tensor], verbose: bool = False
) -> list[float]:
    min_lower_bound = output_bound[0].min().item()
    return [min_lower_bound, min_lower_bound]


def cal_priority_by_max_lower_bound(
    output_bound: tuple[Tensor, Tensor], verbose: bool = False
) -> list[float]:
    max_lower_bound = output_bound[0].max().item()
    return [max_lower_bound, max_lower_bound]


def cal_priority_by_min_distance_zero(
    output_bound: tuple[Tensor, Tensor], verbose: bool = False
) -> list[float]:
    l, u = output_bound
    min_distance = torch.min(torch.abs(l), torch.abs(u)).min().item()
    return [min_distance, min_distance]


def cal_priority_by_max_distance_zero(
    output_bound: tuple[Tensor, Tensor], verbose: bool = False
) -> list[float]:
    l, u = output_bound
    max_distance = torch.max(torch.abs(l), torch.abs(u)).max().item()
    return [max_distance, max_distance]


if __name__ == "__main__":
    l = torch.tensor([[[0.6000, 0.0312, -0.1250, 0.4500, -0.5000]]])
    u = torch.tensor([[[0.6399, 0.0625, -0.0625, 0.5000, -0.4500]]])
    w1 = torch.tensor(
        [
            [[[-0.0098, 0.4945, 0.0172, 0.0490, 0.0320]]],
            [[[-0.0053, 0.3499, 0.0128, 0.0422, 0.0182]]],
            [[[-0.0119, 0.4838, 0.0157, 0.0492, 0.0263]]],
            [[[-0.0036, 0.3984, 0.0124, 0.0515, 0.0076]]],
        ]
    )
    b1 = torch.tensor([-0.0801, -0.0283, -0.0828, -0.0443])
    w2 = torch.tensor(
        [
            [[[-0.0144, -0.2563, -0.0112, -0.0283, 0.0019]]],
            [[[-0.0041, -0.2314, 0.0033, -0.0254, 0.0030]]],
            [[[-0.0194, -0.2266, -0.0242, -0.0222, -0.0114]]],
            [[[-0.0048, -0.2522, 0.0003, -0.0255, -0.0023]]],
        ]
    )
    b2 = torch.tensor([0.0735, 0.0982, 0.0641, 0.0992])
    or_groups = [[0, 1, 2, 3]]
    clamp_intput_bounds_by_lower_output_final_symbnds(
        (l, u), (w1, b1, w2, b2), or_groups
    )
