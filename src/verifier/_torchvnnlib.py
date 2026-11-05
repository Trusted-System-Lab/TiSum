__all__ = ["torch_vnnlib_by_instances", "torch_vnnlib_by_path"]

import os

from src.verifier._utils import TORCHVNNLIB_DIR_NAME
from src.torchvnnlib import TorchVNNLIB


def _torch_vnnlib_model(
    torch_vnnlib: TorchVNNLIB, vnnlib_file_path: str, target_dir_path: str = None
):
    sub_folder_path = os.path.normpath(vnnlib_file_path)
    sub_folder_path = sub_folder_path.replace(
        f"vnnlib{os.sep}", f"{TORCHVNNLIB_DIR_NAME}{os.sep}"
    )
    sub_folder_path = sub_folder_path.replace(".vnnlib", "")
    if target_dir_path is not None:
        sub_folder_path = target_dir_path

    if os.path.isdir(sub_folder_path) and os.listdir(sub_folder_path):
        if torch_vnnlib.verbose:
            print(f"  [CACHE] TorchVNNLIB exists: {sub_folder_path}")
        return

    try:
        torch_vnnlib.convert(vnnlib_file_path, sub_folder_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to convert {vnnlib_file_path} to {sub_folder_path}: \n\t{e}"
        )


def torch_vnnlib_by_instances(
    target_dir_path: str, vnnlib_file_paths: set[str], verbose: bool = False
):
    if not os.path.exists(f"{target_dir_path}/vnnlib"):
        raise ValueError(f"Cannot find vnnlib dir path {target_dir_path}/onnx.")
    if not os.path.exists(f"{target_dir_path}/{TORCHVNNLIB_DIR_NAME}"):
        os.makedirs(f"{target_dir_path}/{TORCHVNNLIB_DIR_NAME}")

    torch_vnnlib = TorchVNNLIB(verbose)

    for vnnlib_file_path in vnnlib_file_paths:
        if not vnnlib_file_path.endswith(".vnnlib"):
            continue

        _torch_vnnlib_model(torch_vnnlib, vnnlib_file_path)

    if verbose:
        print(
            f"Successfully converted {len(vnnlib_file_paths)} vnnlib files to torch "
            f"files in {target_dir_path}/{TORCHVNNLIB_DIR_NAME}."
        )


def torch_vnnlib_by_path(
    vnnlib_file_path: str, target_dir_path: str = None, verbose: bool = False
):
    if not os.path.exists(vnnlib_file_path):
        raise FileNotFoundError(f"Cannot find vnnlib file path {vnnlib_file_path}.")
    if not vnnlib_file_path.endswith(".vnnlib"):
        raise ValueError(f"File {vnnlib_file_path} is not a vnnlib file.")
    vnnlib_dir_name = os.path.basename(vnnlib_file_path).split(".")[:-1]
    vnnlib_dir_name = ".".join(vnnlib_dir_name)
    target_dir_path = os.path.join(target_dir_path, vnnlib_dir_name)

    torch_vnnlib = TorchVNNLIB(verbose)
    _torch_vnnlib_model(torch_vnnlib, vnnlib_file_path, target_dir_path)
