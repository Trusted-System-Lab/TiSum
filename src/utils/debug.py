__doc__ = ["restructuredtext"]
__all__ = ["debug_bound", "debug_rlx", "debug_symbnd"]

import torch

from src.substorch import cal_scalar_bound_lower


def debug_bound(cache, node):
    """You can check the bounds after put the following code at the end of _bwdprop_symbnd from src.utils.debug import debug_bound debug_bound(self.cache, self)."""
    print()
    print("=" * 50)
    print(f"[DEBUG] Check bound of {node.name}")
    from src.substorch import cal_scalar_bound

    for node_name, bound in cache.bnds.items():
        if torch.any(bound[0] > bound[1]):
            print(f"[DEBUG] {node_name} has invalid bound.")
            print(f"[DEBUG] {bound[0]} > {bound[1]}")
            raise ValueError(
                f"[DEBUG] {node_name} has invalid bound. {bound[0]} > {bound[1]}"
            )

    symbnd = node._get_bwdprop_symbnd()
    bound = cache.bnds.get(node.name)
    if bound is None:
        print("[DEBUG] No bound found.")
    else:
        print(f"[DEBUG] symbnd[0].shape={symbnd[0].shape}")
        print(f"[DEBUG] bound[0].shape={bound[0].shape}")
        print("[DEBUG] Symbolic bounds:")
        for s in symbnd:
            print(s[:10])
        print("[DEBUG] Interval bounds:")
        for b in bound:
            print(b[:10])

        if len(symbnd) == 4:
            result = cal_scalar_bound(*symbnd, *bound)
        else:
            result = cal_scalar_bound_lower(*symbnd, *bound)

        print("[DEBUG] Result:")
        for r in result:
            print(r)

        if len(result) == 2:
            if torch.any(result[0] > result[1]):
                print(f"[DEBUG] {node.name} has invalid bound.")
                print(f"[DEBUG] {result[0]} > {result[1]}")
                raise ValueError(
                    f"[DEBUG] {node.name} has invalid bound. {result[0]} > {result[1]}"
                )

    print("=" * 50)
    print()


def debug_rlx(cache, node):
    relaxation = cache.rlxs.get(node.name)
    print()
    print("=" * 50)
    print("[DEBUG] Check relaxation of {node.name}")
    print(f"[DEBUG] {len(relaxation)} elements in relaxation")
    print(relaxation[0][:10])
    print(relaxation[1][:10])
    if len(relaxation) > 2:
        print(relaxation[2][:10])
        print(relaxation[3][:10])
    if len(relaxation) > 4:
        print(relaxation[4][:10])
        print(relaxation[5][:10])
        print(relaxation[6][:10])
        print(relaxation[7][:10])
    print("=" * 50)
    print()


def debug_symbnd(cache, node):
    sybnd = cache.symbnds.get(node.name)
    print()
    print("=" * 50)
    print(f"[DEBUG] Check symbolic bound of {node.name}")
    print(f"[DEBUG] {len(sybnd)} elements in symbolic bound")
    print(sybnd[0][:10])
    print(sybnd[1][:10])
    if len(sybnd) > 2:
        print(sybnd[2][:10])
        print(sybnd[3][:10])
    if len(sybnd) > 4:
        print(sybnd[4][:10])
        print(sybnd[5][:10])
        print(sybnd[6][:10])
        print(sybnd[7][:10])

    print("=" * 50)
    print()
