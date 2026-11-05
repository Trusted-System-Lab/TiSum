__all__ = ["VNNCOMPVerifier"]

import os
import time

import torch

from src.args import BasicArgs
from src.enums import VerifStatus
from src.factory import ModelFactory
from src.utils.slim_kwargs import ALL_TRUE_SLIM_KWARGS
from src.verifier._slimonnx import slim_onnx_by_path
from src.verifier._torchonnx import torch_onnx_by_path
from src.verifier._torchvnnlib import torch_vnnlib_by_path
from src.verifier._utils import (
    flatten_torchvnnlib_dir,
    string_adv_example,
    SLIMONNX_DIR_NAME,
    TORCHONNX_DIR_NAME,
    TORCHVNNLIB_DIR_NAME,
    run_models,
)


class VNNCOMPVerifier:

    def __init__(
        self,
        category: str,
        onnx_file: str,
        vnnlib_file: str,
        result_file: str,
        timeout: float,
        cache_dir: str,
        arguments_list: list[BasicArgs],
        apply_slimonnx: bool = False,
        apply_torchvnnlib: bool = False,
        apply_torchonnx: bool = False,
        verbose: bool = False,
    ):
        self._onnx_path = onnx_file
        self._vnnlib_path = vnnlib_file
        self._result_file = result_file
        self._timeout = timeout
        self._cache_dir = cache_dir
        self._arguments_list = arguments_list
        self.verbose = verbose

        if os.path.dirname(self.result_file) != "":
            os.makedirs(os.path.dirname(self.result_file), exist_ok=True)
        with open(self.result_file, "w") as f:
            f.write("unknown")

        if apply_slimonnx:
            slim_kwargs = ALL_TRUE_SLIM_KWARGS.get(category)
            target_dir_path = os.path.join(cache_dir, SLIMONNX_DIR_NAME)
            os.makedirs(target_dir_path, exist_ok=True)
            slim_onnx_by_path(
                onnx_file,
                slim_kwargs,
                target_dir_path=target_dir_path,
                verbose=verbose,
            )

        if apply_torchvnnlib:
            target_dir_path = os.path.join(cache_dir, TORCHVNNLIB_DIR_NAME)
            os.makedirs(target_dir_path, exist_ok=True)
            torch_vnnlib_by_path(
                vnnlib_file, target_dir_path=target_dir_path, verbose=verbose
            )
        if apply_torchonnx:
            target_dir_path = os.path.join(cache_dir, TORCHONNX_DIR_NAME)
            os.makedirs(target_dir_path, exist_ok=True)
            torch_onnx_by_path(
                onnx_file, target_dir_path=target_dir_path, verbose=verbose
            )

        onnx_file_name = os.path.basename(onnx_file)
        self._slimonnx_path = os.path.join(
            cache_dir,
            SLIMONNX_DIR_NAME,
            onnx_file_name.replace(".onnx", "_slimmed.onnx"),
        )
        self._torchvnnlib_dir = os.path.join(
            cache_dir,
            TORCHVNNLIB_DIR_NAME,
            os.path.basename(vnnlib_file).replace(".vnnlib", ""),
        )
        self._torchonnx_path = os.path.join(
            cache_dir,
            TORCHONNX_DIR_NAME,
            onnx_file_name.replace(".onnx", "_slimmed.py"),
        )
        self._torchonnx_param_path = os.path.join(
            cache_dir,
            TORCHONNX_DIR_NAME,
            onnx_file_name.replace(".onnx", "_slimmed.pth"),
        )
        if not os.path.exists(self._slimonnx_path):
            raise FileNotFoundError(f"ONNX file not found at {self._slimonnx_path}")
        if not os.path.exists(self._vnnlib_path):
            raise FileNotFoundError(f"VNNLIB file not found at {self._vnnlib_path}")
        if not os.path.exists(self._torchvnnlib_dir):
            raise FileNotFoundError(
                f"PyTorch VNNLIB directory not found at {self._torchvnnlib_dir}"
            )
        if not os.path.exists(self._torchonnx_path):
            raise FileNotFoundError(
                f"PyTorch model file not found at {self._torchonnx_path}"
            )
        if not os.path.exists(self._torchonnx_param_path):
            raise FileNotFoundError(
                f"PyTorch parameter file not found at {self._torchonnx_param_path}"
            )

    def verify(self):
        t = time.perf_counter()

        def _early_stop():
            nonlocal has_unsafe, has_all_safe
            if status == VerifStatus.UNSAFE:
                has_unsafe = 1
                has_all_safe = 0
                return True
            elif status == VerifStatus.SAFE:
                if has_all_safe == -1:
                    has_all_safe = 1
            else:
                has_all_safe = 0
            return False

        def _set_group_status():
            nonlocal has_unsafe, has_all_safe
            if has_unsafe == 1:
                return VerifStatus.UNSAFE
            elif has_all_safe == 1:
                return VerifStatus.SAFE
            else:
                return VerifStatus.UNKNOWN

        def _set_final_status():
            nonlocal and_results
            if all(r == VerifStatus.SAFE for r in and_results):
                return VerifStatus.SAFE
            elif any(r == VerifStatus.UNSAFE for r in and_results):
                return VerifStatus.UNSAFE
            else:
                return VerifStatus.UNKNOWN

        properties_paths = flatten_torchvnnlib_dir(self.torchvnnlib_dir)
        for argument in self._arguments_list:
            argument.set_onnx_path(self.onnx_path)
            argument.set_slimonnx_path(self.slimonnx_path)
            argument.set_torchonnx_path(self.torchonnx_path)
        models = ModelFactory.build_models_by_args_list(
            self._arguments_list, self.verbose
        )

        adv_example = None
        and_results = []
        for i, (or_group_path, prop_paths) in enumerate(properties_paths.items()):
            has_unsafe = 0
            has_all_safe = -1
            for j, prop_path in enumerate(prop_paths):
                if self.verbose:
                    print(f"Verifying OR-{i} (Property-{j})...")

                data = torch.load(prop_path, weights_only=True)
                status, adv_example = run_models(
                    models,
                    input_constrs=data["input"],
                    output_constrs=data["output"],
                    verbose=self.verbose,
                )

                if _early_stop():
                    break

            or_result = _set_group_status()
            and_results.append(or_result)

        result = _set_final_status()

        if os.path.dirname(self.result_file) != "":
            os.makedirs(os.path.dirname(self.result_file), exist_ok=True)
        with open(self.result_file, "w") as f:
            if result == VerifStatus.SAFE:
                f.write("unsat")
            elif result == VerifStatus.UNSAFE:
                f.write("sat\n")
                f.write(string_adv_example(*adv_example))
            else:
                f.write("unknown")

        if self.verbose:
            cur_t = time.perf_counter() - t
            print(f"Verification completed in {cur_t:.2f} seconds.")
            print(f"Result for all properties: {and_results}")
            print(f"Final result: {result}")

    @property
    def onnx_path(self):
        return self._onnx_path

    @property
    def slimonnx_path(self) -> str:
        return self._slimonnx_path

    @property
    def vnnlib_path(self) -> str:
        return self._vnnlib_path

    @property
    def result_file(self) -> str:
        return self._result_file

    @property
    def timeout(self) -> float:
        return self._timeout

    @property
    def cache_dir(self) -> str:
        return self._cache_dir

    @property
    def torchvnnlib_dir(self) -> str:
        return self._torchvnnlib_dir

    @property
    def torchonnx_path(self) -> str:
        return self._torchonnx_path

    @property
    def torchonnx_param_path(self) -> str:
        return self._torchonnx_param_path

    @property
    def arguments_list(self) -> list[BasicArgs]:
        return self._arguments_list
