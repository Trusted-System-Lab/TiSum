__all__ = ["store_and_update_bnd", "batched_store_and_update_bnd"]

import torch
from torch import Tensor


def check_bound(
    bound: tuple[Tensor, ...],
    node_name: str,
    new_bound: tuple[Tensor, ...] | None = None,
    old_bound: tuple[Tensor, ...] | None = None,
):
    if len(bound) == 1:
        return
    lower_bound, upper_bound = bound

    tol = 1e-3
    if upper_bound is not None:
        if not torch.all(lower_bound <= upper_bound + tol):
            l = lower_bound.flatten()
            u = upper_bound.flatten()
            n = lower_bound.numel()
            idxs = torch.arange(n, device=lower_bound.device, dtype=torch.long)
            mask = l > u
            msg = (
                f"{node_name} has larger lower bounds than upper bounds:\n"
                f"idxs={idxs[mask].tolist()}\n"
                f"final_l={l[mask].tolist()}\n"
                f"final_u={u[mask].tolist()}\n"
            )
            if old_bound is not None:
                old_lower_bound, old_upper_bound = old_bound
                old_lower_bound = old_lower_bound.flatten()[mask].tolist()
                if old_upper_bound is not None:
                    old_upper_bound = old_upper_bound.flatten()[mask].tolist()
                msg += f"old_l={old_lower_bound}\nold_u={old_upper_bound}\n"
            if new_bound is not None:
                new_lower_bound, new_upper_bound = new_bound
                new_lower_bound = new_lower_bound.flatten()[mask].tolist()
                if new_upper_bound is not None:
                    new_upper_bound = new_upper_bound.flatten()[mask].tolist()
                msg += f"new_l={new_lower_bound}\nnew_u={new_upper_bound}\n"
            msg = msg[:-1]
            raise RuntimeError(msg)

        if not torch.all(lower_bound <= upper_bound):
            new_lower_bound = torch.minimum(lower_bound, upper_bound)
            new_upper_bound = torch.maximum(lower_bound, upper_bound)
            bound = (new_lower_bound, new_upper_bound)

    return bound


def store_and_update_bnd(
    new_bound: tuple[Tensor, ...] | tuple[tuple[Tensor, ...], ...],
    cache: "BasicCache",
    node: "BasicNode",
) -> tuple[Tensor, ...]:
    from src.nodes import OutputNode

    new_bound = check_bound(new_bound, node.name)
    node_name = node.name

    if not isinstance(node, OutputNode):
        assert len(node.output_sizes) == 1, f"{node.output_sizes}"
        bound_shape = node.output_sizes[0]
    else:
        bound_shape = (node.weight.size(1),)

    lower_bound = new_bound[0].reshape(bound_shape)
    if len(new_bound) == 2:
        upper_bound = new_bound[1].reshape(bound_shape)
        new_bound = (lower_bound, upper_bound)
    else:
        new_bound = (lower_bound,)

    old_bound = cache.bnds.get(node_name)

    if old_bound is None:
        cache.bnds[node_name] = new_bound
        return new_bound

    old_lower_bound = old_bound[0]
    old_upper_bound = None if len(old_bound) == 1 else old_bound[1]
    assert old_lower_bound is not None

    new_lower_bound = new_bound[0]
    new_upper_bound = None if len(new_bound) == 1 else new_bound[1]

    final_lower_bound = torch.maximum(new_lower_bound, old_lower_bound)
    final_upper_bound = None
    if old_upper_bound is not None and new_upper_bound is not None:
        final_upper_bound = torch.minimum(new_upper_bound, old_upper_bound)
    elif old_upper_bound is not None:
        final_upper_bound = old_upper_bound

    final_bound = (final_lower_bound, final_upper_bound)
    final_bound = check_bound(final_bound, node.name, new_bound, old_bound)
    cache.bnds[node_name] = final_bound

    return final_bound


def batched_store_and_update_bnd(
    new_bound: tuple[Tensor, ...],
    cache: "BasicCache",
    node: "BasicNode",
) -> tuple[Tensor, ...]:
    from src.nodes import OutputNode

    new_bound = check_bound(new_bound, node.name)
    node_name = node.name

    if not isinstance(node, OutputNode):
        assert len(node.output_sizes) == 1, f"{node.output_sizes}"
        bound_shape = node.output_sizes[0]
    else:
        bound_shape = (node.weight.size(1),)

    batch_size = new_bound[0].size(0)
    lower_bound = new_bound[0].reshape(batch_size, *bound_shape)
    if len(new_bound) == 2:
        upper_bound = new_bound[1].reshape(batch_size, *bound_shape)
        new_bound = (lower_bound, upper_bound)
    else:
        new_bound = (lower_bound,)

    old_bound = cache.bnds.get(node_name)

    if old_bound is None:
        cache.bnds[node_name] = new_bound
        return new_bound

    old_lower_bound = old_bound[0]
    old_upper_bound = None if len(old_bound) == 1 else old_bound[1]
    assert old_lower_bound is not None

    new_lower_bound = new_bound[0]
    new_upper_bound = None if len(new_bound) == 1 else new_bound[1]

    final_lower_bound = torch.maximum(new_lower_bound, old_lower_bound)
    final_upper_bound = None
    if old_upper_bound is not None and new_upper_bound is not None:
        final_upper_bound = torch.minimum(new_upper_bound, old_upper_bound)
    elif old_upper_bound is not None:
        final_upper_bound = old_upper_bound

    final_bound = (final_lower_bound, final_upper_bound)
    final_bound = check_bound(final_bound, node.name, new_bound, old_bound)
    cache.bnds[node_name] = final_bound

    return final_bound
