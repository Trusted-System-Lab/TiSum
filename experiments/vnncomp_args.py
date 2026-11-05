__docformat__ = ["restructuredtext"]
__all__ = ["get_arguments_list"]

import os.path
from typing import Any

import yaml

from src.args import *

ARGS_DICT: dict[str, Any] = {
    "OriModelArgs": OriModelArgs,
    "AdvAttackArgs": AdvAttackArgs,
    "LinPropArgs": LinPropArgs,
    "AdaPropArgs": AdaPropArgs,
    "InputSplitPropArgs": InputSplitPropArgs,
    "BatchedInputSplitPropArgs": BatchedInputSplitPropArgs,
    "QuadPropArgs": QuadPropArgs,
    "LinProgArgs": LinProgArgs,
}


def get_arguments_list(
    category: str, shared_kwargs: dict[str, Any], suffix: str = ""
) -> list[BasicArgs]:
    # Read the configuration file named "configs/{category}.yaml".
    # Get the dictionary of this file.
    dir_name = os.path.dirname(__file__)
    dir_name = os.path.join(dir_name, "configs")
    config_file_path = os.path.join(dir_name, f"{category}{suffix}.yaml")
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(
            f"The configuration file '{config_file_path}' does not exist."
        )
    with open(config_file_path, "r") as file:
        try:
            args_dict = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    arguments_list = []
    for args_name, kwargs in args_dict.items():
        if args_name not in ARGS_DICT:
            raise ValueError(f"Unknown argument class: {args_name}")
        # Create an instance of the argument class with the provided kwargs.
        argument_instance = ARGS_DICT[args_name](**kwargs)
        arguments_list.append(argument_instance)

    for argument in arguments_list:
        for key, value in shared_kwargs.items():
            object.__setattr__(argument, key, value)
        argument.__post_init__()  # Call the post-init method to finalize the setup

    return arguments_list
