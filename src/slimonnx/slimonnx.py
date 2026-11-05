__all__ = ["SlimONNX"]

import os.path
import time

import onnx

from .optimize_onnx import optimize_onnx


class SlimONNX:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def slim(
        self,
        onnx_path: str,
        target_path: str | None = None,
        constant_to_initializer: bool = True,
        fuse_constant_nodes: bool = False,
        fuse_matmul_add: bool = False,
        fuse_gemm_reshape_bn: bool = False,
        fuse_bn_reshape_gemm: bool = False,
        fuse_bn_gemm: bool = False,
        fuse_transpose_bn_transpose: bool = False,
        fuse_gemm_gemm: bool = False,
        fuse_conv_bn: bool = False,
        fuse_bn_conv: bool = False,
        fuse_convtransposed_bn: bool = False,
        fuse_bn_convtransposed: bool = False,
        simplify_conv_to_flatten_gemm: bool = False,
        simplify_gemm: bool = True,
        remove_redundant_operations: bool = False,
        simplify_node_name: bool = True,
        reorder_by_strict_topological_order: bool = True,
        validate_model: bool = True,
    ):
        """Simplify the ONNX model by fusing some nodes."""
        if self.verbose:
            print(f"Slim ONNX model {onnx_path}...")
            t = time.perf_counter()
        model = onnx.load(onnx_path)

        new_model = optimize_onnx(
            model,
            constant_to_initializer=constant_to_initializer,
            fuse_constant_nodes=fuse_constant_nodes,
            fuse_matmul_add=fuse_matmul_add,
            fuse_gemm_reshape_bn=fuse_gemm_reshape_bn,
            fuse_bn_reshape_gemm=fuse_bn_reshape_gemm,
            fuse_bn_gemm=fuse_bn_gemm,
            fuse_transpose_bn_transpose=fuse_transpose_bn_transpose,
            fuse_gemm_gemm=fuse_gemm_gemm,
            fuse_conv_bn=fuse_conv_bn,
            fuse_bn_conv=fuse_bn_conv,
            fuse_convtransposed_bn=fuse_convtransposed_bn,
            fuse_bn_convtransposed=fuse_bn_convtransposed,
            simplify_conv_to_flatten_gemm=simplify_conv_to_flatten_gemm,
            simplify_gemm=simplify_gemm,
            remove_redundant_operations=remove_redundant_operations,
            reorder_by_strict_topological_order=reorder_by_strict_topological_order,
            simplify_node_name=simplify_node_name,
            verbose=self.verbose,
        )

        if target_path is None:
            target_path = onnx_path.replace(".onnx", "_simplified.onnx")

        if not os.path.exists(os.path.dirname(target_path)):
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

        onnx.save(new_model, target_path)

        if self.verbose:
            t = time.perf_counter() - t
            print(f"Slimmed ONNX model saved to {target_path} ({t:.4f}s)")
