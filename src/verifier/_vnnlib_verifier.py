__all__ = ["VNNLIBVerifier"]

import os
import time
from typing import Any

import torch

from src.args import BasicArgs
from src.enums import VerifStatus
from src.factory import ModelFactory
from src.verifier._slimonnx import slim_onnx_by_vnnlib_instances
from src.verifier._torchonnx import torch_onnx_by_instances
from src.verifier._torchvnnlib import torch_vnnlib_by_instances
from src.verifier._utils import (
    INSTANCES_FILE_NAME,
    SLIMONNX_DIR_NAME,
    TORCHONNX_DIR_NAME,
    TORCHVNNLIB_DIR_NAME,
    get_deepest_two_levels,
    flatten_torchvnnlib_dir,
    string_adv_example,
    run_models,
)


def _read_original_instances(
    instances_file_path: str, target_dir_path: str
) -> list[tuple[str, str, float]]:
    with open(instances_file_path, "r") as f:
        lines = f.readlines()

    instances = []
    for line in lines:
        line = line.replace("\n", "").strip()
        model_path, vnnlib_path, timeout = line.split(",")
        model_path = os.path.join(target_dir_path, model_path)
        vnnlib_path = os.path.join(target_dir_path, vnnlib_path)
        model_path = os.path.normpath(model_path)
        vnnlib_path = os.path.normpath(vnnlib_path)
        timeout = float(timeout)
        instances.append((model_path, vnnlib_path, timeout))

    return instances


def _get_rover_instances(
    instances: list[tuple[str, str, float]],
) -> list[dict[str, str]]:
    """This just get the paths without checking the existence of the files."""
    rover_instances = []
    for instance in instances:
        model_path, vnnlib_path, timeout = instance
        model_path = os.path.normpath(model_path)
        vnnlib_path = os.path.normpath(vnnlib_path)
        timeout = float(timeout)

        onnx_path = model_path
        torchonnx_path = model_path.replace(
            f"onnx{os.sep}", f"{TORCHONNX_DIR_NAME}" f"{os.sep}"
        ).replace(".onnx", "_slimmed.py")
        slimonnx_path = model_path.replace(
            f"onnx{os.sep}", f"{SLIMONNX_DIR_NAME}" f"{os.sep}"
        ).replace(".onnx", "_slimmed.onnx")

        torchvnnlib_path = vnnlib_path.replace(
            f"vnnlib{os.sep}", f"{TORCHVNNLIB_DIR_NAME}" f"{os.sep}"
        ).replace(".vnnlib", "")

        rover_instances.append(
            {
                "onnx": onnx_path,
                "torchonnx": torchonnx_path,
                "slimonnx": slimonnx_path,
                "torchvnnlib": torchvnnlib_path,
                "timeout": timeout,
            }
        )


    return rover_instances


def _parse_one_instance(
    instance_idx: int, instance: dict[str, str], verbose: bool = False
) -> dict[str, Any]:
    onnx_path = instance["onnx"]
    torchonnx_path = instance["torchonnx"]
    slimonnx_path = instance["slimonnx"]
    torchvnnlib_folder = instance["torchvnnlib"]
    timeout = instance["timeout"]

    if verbose:
        short_torchonnx_path = get_deepest_two_levels(torchonnx_path)
        short_slimonnx_path = get_deepest_two_levels(slimonnx_path)
        short_torchvnnlib_folder = get_deepest_two_levels(torchvnnlib_folder)
        print(
            f"Instance-{instance_idx}:\n"
            f"\ttorchonnx_path={short_torchonnx_path}\n"
            f"\tslimonnx_path={short_slimonnx_path}\n"
            f"\ttorchvnnlib_folder={short_torchvnnlib_folder}\n"
            f"\ttimeout={timeout} s"
        )

    all_properties = flatten_torchvnnlib_dir(torchvnnlib_folder, verbose)

    instance = {
        "onnx": onnx_path,
        "torchonnx": torchonnx_path,
        "slimonnx": slimonnx_path,
        "torchvnnlib": torchvnnlib_folder,
        "timeout": timeout,
        "properties": all_properties,
    }

    return instance


def create_results_file(
    target_dir_path: str,
    results_dir: str = None,
    results_timestamp: str = None,
) -> str:
    instances_csv_path = os.path.join(target_dir_path, INSTANCES_FILE_NAME)
    instances_csv_path = os.path.normpath(instances_csv_path)

    target_dir_name = os.path.basename(target_dir_path)
    if results_dir is None:
        results_dir = "./.temp/results"
    if results_timestamp is None:
        results_timestamp = ""
    else:
        results_timestamp = "_" + results_timestamp

    results_instances_csv_path = os.path.join(
        results_dir, target_dir_name, f"results{results_timestamp}.csv"
    )

    os.makedirs(os.path.dirname(results_instances_csv_path), exist_ok=True)
    with open(instances_csv_path, "r") as src_file:
        with open(results_instances_csv_path, "w") as dst_file:
            dst_file.write(src_file.read())

    return results_instances_csv_path


def update_result_in_file(
    results_file_path: str, instance_idx: int, result_value: str, sample_time: float
):
    results_file_path = os.path.normpath(results_file_path)

    with open(results_file_path, "r") as f:
        lines = f.readlines()

    if not (0 <= instance_idx < len(lines)):
        raise IndexError(
            f"Instance index {instance_idx} is out of range for results file '{results_file_path}'."
        )

    parts = lines[instance_idx].strip().split(",")

    while len(parts) < 5:
        parts.append("")

    parts[3] = str(result_value)
    parts[4] = f"{sample_time:.4f}"

    lines[instance_idx] = ",".join(parts) + "\n"

    with open(results_file_path, "w") as f:
        f.writelines(lines)


class VNNLIBVerifier:
    _instances: list[dict[str, str]]
    _arguments_list: list[BasicArgs]
    verbose: bool

    def __init__(
        self,
        target_dir_path: str,
        arguments_list: list[BasicArgs],
        apply_slimonnx: bool = False,
        apply_torchvnnlib: bool = False,
        apply_torchonnx: bool = False,
        specified_instances: list[int] = None,
        record_result: bool = False,
        results_instances_csv_dir: str = None,
        results_timestamp: str = None,
        verbose: bool = False,
    ):
        self._arguments_list = arguments_list
        self._specified_instances = specified_instances
        self.verbose = verbose
        if not os.path.exists(target_dir_path):
            raise ValueError(f"Cannot find vnnlib dir path {target_dir_path}.")

        instances_file_path = os.path.join(target_dir_path, INSTANCES_FILE_NAME)
        instances_file_path = os.path.normpath(instances_file_path)
        if not os.path.exists(instances_file_path):
            raise ValueError(f"Cannot find instances file {instances_file_path}.")
        instances = _read_original_instances(instances_file_path, target_dir_path)
        if self.verbose:
            print(f"Read {len(instances)} instances from {instances_file_path}.")

        onnx_file_paths = set()
        vnnlib_data_file_paths = set()
        for instance in instances:
            onnx_file_path = instance[0]
            vnnlib_data_file_path = instance[1]
            onnx_file_paths.add(onnx_file_path)
            vnnlib_data_file_paths.add(vnnlib_data_file_path)

        if apply_slimonnx:
            t = time.perf_counter()
            slim_onnx_by_vnnlib_instances(
                target_dir_path, onnx_file_paths, verbose=verbose
            )
            if self.verbose:
                print(f"Slim ONNX files in {time.perf_counter() - t:.2f}s.")
        if apply_torchvnnlib:
            t = time.perf_counter()
            torch_vnnlib_by_instances(target_dir_path, vnnlib_data_file_paths, verbose)
            if self.verbose:
                print(f"Torch VNNLIB files in {time.perf_counter() - t:.2f}s.")
        slimonnx_dir_path = os.path.join(target_dir_path, SLIMONNX_DIR_NAME)
        slimonnx_dir_path = os.path.normpath(slimonnx_dir_path)
        if not os.path.exists(slimonnx_dir_path):
            raise ValueError(f"Cannot find onnx dir path {slimonnx_dir_path}.")
        torchvnnlib_dir_path = os.path.join(target_dir_path, TORCHVNNLIB_DIR_NAME)
        torchvnnlib_dir_path = os.path.normpath(torchvnnlib_dir_path)
        if not os.path.exists(torchvnnlib_dir_path):
            raise ValueError(f"Cannot find vnnlib dir path {torchvnnlib_dir_path}.")

        self._instances = _get_rover_instances(instances)
        onnx_file_paths = set()
        for instance in self._instances:
            onnx_file_paths.add(instance["slimonnx"])

        if apply_torchonnx:
            t = time.perf_counter()
            torch_onnx_by_instances(target_dir_path, onnx_file_paths, verbose)
            if self.verbose:
                print(f"Torch ONNX files in {time.perf_counter() - t:.2f}s.")
        torchonnx_dir_path = os.path.join(target_dir_path, TORCHONNX_DIR_NAME)
        torchonnx_dir_path = os.path.normpath(torchonnx_dir_path)
        if not os.path.exists(torchonnx_dir_path):
            raise ValueError(f"Cannot find torchonnx dir path {torchonnx_dir_path}.")

        self._results_file_path = None
        if record_result:
            self._results_file_path = create_results_file(
                target_dir_path, results_instances_csv_dir, results_timestamp
            )

    def verify(self):
        t = time.perf_counter()
        if self.verbose:
            print(f"Start verifying {len(self._instances)} instances...")
        results_stats: dict[VerifStatus, int] = {
            VerifStatus.SAFE: 0,
            VerifStatus.UNSAFE: 0,
            VerifStatus.UNKNOWN: 0,
        }
        instance_stats: dict[int, tuple[VerifStatus, float]] = {}

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

        n_instance = 0
        n_models = 0
        n_prop = 0
        pre_model_path = None
        models = None
        instance_idxs = self.specified_instances
        for instance_idx, instance in enumerate(self._instances):
            if instance_idxs is not None and instance_idx not in instance_idxs:
                continue
            t_instance = time.perf_counter()
            n_instance += 1

            instance = _parse_one_instance(instance_idx, instance, self.verbose)

            onnx_path = instance["onnx"]
            slimonnx_path = instance["slimonnx"]
            torchonnx_path = instance["torchonnx"]
            properties_paths = instance["properties"]
            if pre_model_path != slimonnx_path:
                for argument in self._arguments_list:
                    argument.set_onnx_path(onnx_path)
                    argument.set_slimonnx_path(slimonnx_path)
                    argument.set_torchonnx_path(torchonnx_path)
                models = ModelFactory.build_models_by_args_list(
                    self._arguments_list, self.verbose
                )
                n_models += 1

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
                    n_prop += 1

                    if _early_stop():
                        break

                or_result = _set_group_status()
                and_results.append(or_result)

            result = _set_final_status()
            results_stats[result] += 1
            sample_time = time.perf_counter() - t_instance
            instance_stats[instance_idx] = (result, sample_time)

            pre_model_path = slimonnx_path

            if self._results_file_path is not None:
                result_value = "unknown"
                if result == VerifStatus.SAFE:
                    result_value = "unsat"
                elif result == VerifStatus.UNSAFE:
                    result_value = "sat"
                update_result_in_file(
                    self._results_file_path, instance_idx, result_value, sample_time
                )
                if adv_example is not None:
                    adv_example_file_path = os.path.join(
                        os.path.dirname(self._results_file_path),
                        f"instance_{instance_idx}.counterexample.text",
                    )
                    adv_example_file_path = os.path.normpath(adv_example_file_path)
                    with open(adv_example_file_path, "w") as f:
                        f.write(string_adv_example(*adv_example))

            if self.verbose:
                cur_t = time.perf_counter() - t
                print(f"Results Summary: ")
                for k, v in results_stats.items():
                    print(f"    {str(k):<8}: {v:>5}")
                print(
                    f"Total instances     : {len(self._instances):>3}  \n"
                    f"Processed models    : {n_models:>3}  \n"
                    f"Processed properties: {n_prop:>3}  \n"
                    f"Sample used time    : {sample_time:>6.2f} s\n"
                    f"Total used time     : {cur_t:>6.2f} s\n"
                    f"Average used time   : {cur_t / n_instance:>6.2f} s\n"
                )

    @property
    def instances(self) -> list[dict[str, str]]:
        return self._instances

    @property
    def arguments_list(self) -> list[BasicArgs]:
        return self._arguments_list

    @property
    def specified_instances(self) -> list[int] | None:
        return self._specified_instances

    @property
    def results_file_path(self) -> str | None:
        return self._results_file_path
