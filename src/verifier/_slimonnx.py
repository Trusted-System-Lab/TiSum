__all__ = ["slim_onnx_by_vnnlib_instances", "slim_onnx_by_path"]

import os
import warnings

import onnx

from src.utils.slim_kwargs import ALL_TRUE_SLIM_KWARGS, SLIM_KWARGS
from src.verifier._utils import SLIMONNX_DIR_NAME
from src.slimonnx import SlimONNX


def _slim_onnx_model(
    slim_onnx: SlimONNX,
    slim_kwargs: dict,
    onnx_file_path: str,
    target_dir_path: str = None,
):
    model = onnx.load(onnx_file_path)
    model = onnx.version_converter.convert_version(model, target_version=21)
    onnx_file_path_v22 = onnx_file_path.replace(".onnx", "_v21.onnx")
    onnx.save(model, onnx_file_path_v22)

    onnx_file_path = os.path.normpath(onnx_file_path)
    target_file_path = onnx_file_path.replace(
        f"onnx{os.sep}", f"{SLIMONNX_DIR_NAME}{os.sep}"
    )
    target_file_path = target_file_path.replace(".onnx", "_slimmed.onnx")

    if target_dir_path is not None:
        target_file_path = os.path.join(
            target_dir_path, os.path.basename(target_file_path)
        )
        if not os.path.exists(target_dir_path):
            os.makedirs(target_dir_path)

    if os.path.exists(target_file_path):
        if slim_onnx.verbose:
            print(f"  [CACHE] SlimONNX exists: {target_file_path}")
        os.remove(onnx_file_path_v22)
        return

    try:
        slim_onnx.slim(onnx_file_path_v22, target_file_path, **slim_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to slim {onnx_file_path} to {target_file_path}: \n\t{e}"
        )
    finally:
        os.remove(onnx_file_path_v22)


def slim_onnx_by_vnnlib_instances(
    target_dir_path: str, onnx_file_paths: set[str], verbose: bool = False
):
    if not os.path.exists(f"{target_dir_path}/onnx"):
        raise ValueError(f"Cannot find onnx dir path {target_dir_path}.")
    if not os.path.exists(f"{target_dir_path}/{SLIMONNX_DIR_NAME}"):
        os.makedirs(f"{target_dir_path}/{SLIMONNX_DIR_NAME}")

    benchmark_name = os.path.basename(os.path.normpath(target_dir_path))
    slim_kwargs = SLIM_KWARGS.get(benchmark_name)
    if slim_kwargs is None:
        warnings.warn(
            f"No specific slim kwargs found for this benchmark ({benchmark_name}). "
            f"Using all True and this maybe result in fail because some unsupported "
            f"cases."
        )
        slim_kwargs = ALL_TRUE_SLIM_KWARGS

    slim_onnx = SlimONNX(verbose)
    for onnx_file_path in onnx_file_paths:
        if not onnx_file_path.endswith(".onnx"):
            continue

        _slim_onnx_model(slim_onnx, slim_kwargs, onnx_file_path)

    if verbose:
        print(
            f"Successfully slimmed {len(onnx_file_paths)} onnx files to slimonnx "
            f"files in {target_dir_path}/{SLIMONNX_DIR_NAME}"
        )


def slim_onnx_by_path(
    onnx_file_path: str,
    slim_kwargs: dict = ALL_TRUE_SLIM_KWARGS,
    target_dir_path: str = None,
    verbose: bool = False,
):
    if not os.path.exists(onnx_file_path):
        raise FileNotFoundError(f"Target net path {onnx_file_path} does not exist.")
    if not onnx_file_path.endswith(".onnx"):
        raise ValueError(f"Target net path must be an ONNX file, got {onnx_file_path}.")

    slim_onnx = SlimONNX(verbose)
    _slim_onnx_model(slim_onnx, slim_kwargs, onnx_file_path, target_dir_path)
