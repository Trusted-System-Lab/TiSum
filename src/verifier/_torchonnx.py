__all__ = ["torch_onnx_by_instances", "torch_onnx_by_path"]

import os

from src.torchonnx import TorchONNX
from ._utils import TORCHONNX_DIR_NAME


def _torch_onnx_model(
    torch_onnx: TorchONNX,
    onnx_file_path: str,
    target_dir_path: str = None,
):
    if target_dir_path is None:
        target_dir_path = os.path.dirname(onnx_file_path)
    onnx_file_name = os.path.basename(onnx_file_path)
    py_file_name = onnx_file_name.replace(".onnx", ".py")
    pth_file_name = onnx_file_name.replace(".onnx", ".pth")
    pytorch_model_path = os.path.join(target_dir_path, py_file_name)
    pytorch_param_path = os.path.join(target_dir_path, pth_file_name)

    if os.path.exists(pytorch_model_path) and os.path.exists(pytorch_param_path):
        if torch_onnx.verbose:
            print(f"  [CACHE] TorchONNX exists: {pytorch_model_path}")
        return

    try:
        torch_onnx.convert(
            onnx_file_path,
            target_py_path=pytorch_model_path,
            target_pth_path=pytorch_param_path,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to torch {onnx_file_path}: \n\t{e}")


def torch_onnx_by_instances(
    target_dir_path: str, onnx_file_paths: set[str], verbose: bool = False
):
    if not os.path.exists(f"{target_dir_path}/onnx"):
        raise ValueError(f"Cannot find onnx dir path {target_dir_path}.")
    if not os.path.exists(f"{target_dir_path}/{TORCHONNX_DIR_NAME}"):
        os.makedirs(f"{target_dir_path}/{TORCHONNX_DIR_NAME}")

    torch_onnx = TorchONNX(verbose)
    for onnx_file_path in onnx_file_paths:
        if not onnx_file_path.endswith(".onnx"):
            continue
        onnx_file_path = os.path.normpath(onnx_file_path)
        target_dir_path = os.path.dirname(onnx_file_path)
        target_dir_path = target_dir_path.replace("_slim", "")
        target_dir_path = target_dir_path.replace("onnx", TORCHONNX_DIR_NAME)
        onnx_file_path = os.path.normpath(onnx_file_path)
        _torch_onnx_model(torch_onnx, onnx_file_path, target_dir_path)

    if verbose:
        print(
            f"Successfully converted {len(onnx_file_paths)} onnx files to torch "
            f"files in {target_dir_path}/{TORCHONNX_DIR_NAME}"
        )


def torch_onnx_by_path(
    onnx_file_path: str, target_dir_path: str = None, verbose: bool = False
):
    if not os.path.exists(onnx_file_path):
        raise FileNotFoundError(f"ONNX file '{onnx_file_path}' does not exist.")
    if not onnx_file_path.endswith(".onnx"):
        raise ValueError(
            f"Target net path must be an ONNX file, got '{onnx_file_path}'."
        )
    torch_onnx = TorchONNX(verbose)
    _torch_onnx_model(torch_onnx, onnx_file_path, target_dir_path)
