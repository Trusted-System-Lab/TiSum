__all__ = ["batched_quad_backsub_gemm", "batched_quad_backsub_gemm_lower"]

import torch

from ..quad import quad_backsub_gemm, quad_backsub_gemm_lower

batched_quad_backsub_gemm = torch.vmap(
    quad_backsub_gemm,
    in_dims=(0, 0, 0, 0, 0, 0, 0, 0, None, None),
    out_dims=(0, 0, 0, 0),
)

batched_quad_backsub_gemm_lower = torch.vmap(
    quad_backsub_gemm_lower,
    in_dims=(0, 0, 0, 0, 0, None, None),
    out_dims=(0, 0),
)
