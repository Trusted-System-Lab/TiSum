__all__ = [
    "SLIMONNX_DIR_NAME",
    "TORCHVNNLIB_DIR_NAME",
    "TORCHONNX_DIR_NAME",
    "INSTANCES_FILE_NAME",
    "get_deepest_two_levels",
    "flatten_torchvnnlib_dir",
    "string_adv_example",
    "run_models",
    "run_models_with_same_bounds",
    "batched_run_models_with_same_bounds",
]
import os
import warnings

import torch
from torch import Tensor

from src.cache import LinPropCache
from src.enums import VerifStatus
from src.model import *

SLIMONNX_DIR_NAME = "onnx_slim"
TORCHONNX_DIR_NAME = "onnx_torch"
TORCHVNNLIB_DIR_NAME = "vnnlib_torch"
INSTANCES_FILE_NAME = "instances.csv"


def get_deepest_two_levels(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) >= 2:
        parts = parts[-2:]
    return str(os.path.join(*parts))


def flatten_torchvnnlib_dir(
    torchvnnlib_dir_path: str, verbose: bool = False
) -> dict[str, list[str]]:
    """Flattens the directory structure of a torchvnnlib directory."""
    if not os.path.exists(torchvnnlib_dir_path):
        raise FileNotFoundError(f"Directory not found: {torchvnnlib_dir_path}")

    or_group_paths = [
        os.path.normpath(os.path.join(torchvnnlib_dir_path, sub_folder))
        for sub_folder in os.listdir(torchvnnlib_dir_path)
    ]
    if verbose:
        print(f"Get {len(or_group_paths)} OR groups of properties")

    all_properties = {}
    for or_group_path in or_group_paths:
        and_properties = [
            os.path.normpath(os.path.join(or_group_path, sub_folder))
            for sub_folder in os.listdir(or_group_path)
        ]
        if verbose:
            short_or_group_path = get_deepest_two_levels(or_group_path)
            print(f"Get {len(and_properties)} AND properties for {short_or_group_path}")

        all_properties[or_group_path] = and_properties

    return all_properties


def string_adv_example(x: Tensor, y: Tensor) -> str:
    x = x[0].flatten()
    y = y[0].flatten()
    string = f"((X_{0} {x[0].item()})\n"
    for i in range(1, x.numel()):
        string += f" (X_{i} {x[i].item()})\n"
    for i in range(y.numel()):
        string += f" (Y_{i} {y[i].item()})\n"
    string = string[:-1] + ")"

    return string


def unify_remaining_models_status(
    models: list,
    current_model: BasicModel | OriModel | AdvAttackModel,
    current_model_idx: int,
    verbose: bool,
):
    if current_model_idx >= len(models) - 1:
        return

    if verbose:
        print(f"Unifying all models' status to {type(current_model).__name__}.")

    for model in models[current_model_idx + 1 :]:
        if hasattr(current_model, "cache") and hasattr(model, "cache"):
            model.cache.inherit(current_model.cache)
        model._or_groups = current_model.or_groups
        model.each_prop_status = current_model.each_prop_status


def run_mode_with_memory_handling(
    model: LinPropModel | OriModel | AdvAttackModel,
    input_constrs: Tensor,
    output_constrs: list[Tensor],
):
    try:
        model.run(input_constrs, output_constrs)
    except MemoryError as e:
        print(f"MemoryError occurred in {type(model).__name__}")
        print("Skip this model due to memory constraints.")
        warnings.warn(f"{e}.")
    except torch.cuda.OutOfMemoryError as e:
        print(f"torch.cuda.OutOfMemoryError occurred in {type(model).__name__}")
        print("Skip this model due to GPU memory constraints.")
        warnings.warn(f"{e}.")


def run_models(
    models: list[LinPropModel | OriModel | AdvAttackModel],
    input_constrs: Tensor,
    output_constrs: list[Tensor],
    clear_cache: bool = True,
    keep_bounds: bool = False,
    keep_rlxs: bool = False,
    keep_symbnds: bool = False,
    early_stop: bool = True,
    handle_none_adv_example_by_attack: bool = False,
    verbose: bool = False,
) -> tuple[VerifStatus, tuple[Tensor, Tensor] | None]:
    status = VerifStatus.UNKNOWN

    for model in models:
        model.each_prop_status = None
        model._or_groups = None
        if hasattr(model, "cache"):
            model.cache.clear(
                keep_bnds=keep_bounds,
                keep_rlxs=keep_rlxs,
                keep_symbnds=keep_symbnds,
            )

    adv_example = None
    pre_cache: LinPropCache | None = None
    for i, model in enumerate(models):
        if verbose:
            print(f"\nRunning {type(model).__name__} ({i + 1}/{len(models)})...")

        cache = model.cache if hasattr(model, "cache") else None
        if pre_cache is not None:
            if cache is not None:
                cache.inherit(pre_cache)
            pre_cache.clear()

        run_mode_with_memory_handling(model, input_constrs, output_constrs)

        pre_cache = cache
        status = analyze_status(model.each_prop_status)

        if early_stop:
            if status == VerifStatus.UNSAFE:
                adv_example = model.adv_example
                if adv_example is None and handle_none_adv_example_by_attack:
                    adv_example = try_to_find_adv_example_by_after_attack()
                    if adv_example is None:
                        model.status = VerifStatus.UNKNOWN
                        status = VerifStatus.UNKNOWN
            if status in {VerifStatus.SAFE, VerifStatus.UNSAFE}:
                unify_remaining_models_status(models, model, i, verbose)
                break

    if verbose:
        if pre_cache is not None:
            output_bounds = pre_cache.bnds[model.nodes[-1].name]
            print(f"[COLLECT] Final output bounds:")
            print(f"\t{output_bounds[0].tolist()}")
            print(f"\t{output_bounds[1].tolist()}")

    if pre_cache is not None and clear_cache:
        pre_cache.clear()

    return status, adv_example


def run_models_with_same_bounds(
    models: list[LinPropModel | OriModel | AdvAttackModel],
    input_constrs: Tensor,
    output_constrs: list[Tensor],
    verbose: bool = False,
) -> tuple[VerifStatus, tuple[Tensor, Tensor] | None]:
    """This is used for BaB mode, where all models share the same input/output bounds cache, especially the bounds."""
    for m in models:
        if hasattr(m, "cache"):
            assert id(m.cache.bnds) == id(models[0].cache.bnds)

    for model in models:
        model.each_prop_status = None
        model._or_groups = None
        if hasattr(model, "cache"):
            model.cache.clear(keep_bnds=True)

    status = VerifStatus.UNKNOWN
    adv_example = None
    pre_cache: LinPropCache | None = None
    for i, model in enumerate(models):
        if verbose:
            print(f"\nRunning {type(model).__name__} ({i + 1}/{len(models)})...")

        cache = model.cache if hasattr(model, "cache") else None
        if pre_cache is not None:
            if cache is not None:
                cache.inherit(pre_cache)
            pre_cache.clear(keep_bnds=True, keep_output_final_symbnds=True)

        run_mode_with_memory_handling(model, input_constrs, output_constrs)

        pre_cache = cache
        status = analyze_status(model.each_prop_status)

        if status == VerifStatus.UNSAFE:
            adv_example = model.adv_example
        if status in {VerifStatus.SAFE, VerifStatus.UNSAFE}:
            unify_remaining_models_status(models, model, i, verbose)
            break

    if pre_cache is not None:
        pre_cache.clear(keep_bnds=True, keep_output_final_symbnds=True)

    return status, adv_example


def batched_run_models_with_same_bounds(
    models: list[LinPropModel | OriModel | AdvAttackModel],
    input_constrs: Tensor,
    output_constrs: list[Tensor],
    verbose: bool = False,
):
    for m in models:
        if hasattr(m, "cache"):
            assert id(m.cache.bnds) == id(models[0].cache.bnds)

    for model in models:
        model.each_prop_status = None
        model._or_groups = None
        if hasattr(model, "cache"):
            model.cache.clear(keep_bnds=True)

    pre_cache: LinPropCache | None = None
    for i, model in enumerate(models):
        if verbose:
            print(f"\nRunning {type(model).__name__} ({i + 1}/{len(models)})...")

        cache = model.cache if hasattr(model, "cache") else None
        if pre_cache is not None:
            if cache is not None:
                cache.inherit(pre_cache)
            pre_cache.clear(keep_bnds=True, keep_output_final_symbnds=True)

        run_mode_with_memory_handling(model, input_constrs, output_constrs)

        pre_cache = cache

    if pre_cache is not None:
        pre_cache.clear(keep_bnds=True, keep_output_final_symbnds=True)


def try_to_find_adv_example_by_after_attack():
    raise NotImplementedError
